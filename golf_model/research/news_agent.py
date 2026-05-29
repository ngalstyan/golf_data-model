# ==============================================================================
# golf_model/research/news_agent.py
# ==============================================================================
#
# READ-ONLY NEWS INTELLIGENCE LAYER
# -----------------------------------
# Researches recent news about players in identified H2H matchup bets.
# Produces structured briefings for human review before placing bets.
#
# This module NEVER modifies bets, probabilities, or sizing decisions.
# It is purely informational — the human makes all final calls.
#
# Usage (from notebook 09):
#   from research.news_agent import NewsResearchAgent
#   agent = NewsResearchAgent(settings=cfg)
#   report = agent.research_bets(bets_df, tournament_name, event_id)
#   agent.display_report(report)
#
# ==============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree
import json
import logging
import re
import time
import urllib.parse

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


@dataclass
class NewsArticle:
    """A single news article/snippet found about a player."""
    title: str
    source: str
    url: str
    published: Optional[str] = None
    snippet: str = ""


@dataclass
class PlayerBriefing:
    """Structured research output for a single player."""
    player_name: str
    player_id: int
    articles_found: int
    injury_status: str = "No reports found"
    equipment_changes: str = "No reports found"
    recent_form_commentary: str = "No reports found"
    personal_events: str = "No reports found"
    course_history_comments: str = "No reports found"
    sentiment: Sentiment = Sentiment.UNKNOWN
    key_quotes: List[str] = field(default_factory=list)
    confidence: str = "low"  # none / low / medium / high
    raw_articles: List[NewsArticle] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "player_id": self.player_id,
            "articles_found": self.articles_found,
            "injury_status": self.injury_status,
            "equipment_changes": self.equipment_changes,
            "recent_form_commentary": self.recent_form_commentary,
            "personal_events": self.personal_events,
            "course_history_comments": self.course_history_comments,
            "sentiment": self.sentiment.value,
            "key_quotes": self.key_quotes,
            "confidence": self.confidence,
            "raw_articles": [
                {"title": a.title, "source": a.source, "url": a.url,
                 "published": a.published, "snippet": a.snippet}
                for a in self.raw_articles
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerBriefing":
        articles = [NewsArticle(**a) for a in d.get("raw_articles", [])]
        return cls(
            player_name=d["player_name"],
            player_id=d["player_id"],
            articles_found=d["articles_found"],
            injury_status=d.get("injury_status", "No reports found"),
            equipment_changes=d.get("equipment_changes", "No reports found"),
            recent_form_commentary=d.get("recent_form_commentary", "No reports found"),
            personal_events=d.get("personal_events", "No reports found"),
            course_history_comments=d.get("course_history_comments", "No reports found"),
            sentiment=Sentiment(d.get("sentiment", "unknown")),
            key_quotes=d.get("key_quotes", []),
            confidence=d.get("confidence", "low"),
            raw_articles=articles,
        )


@dataclass
class MatchupBriefing:
    """Complete research report for one H2H matchup."""
    matchup: str
    bet_on: str
    edge: float
    stake: float
    odds: float
    player_a: PlayerBriefing
    player_b: PlayerBriefing
    head_to_head_notes: str = ""


@dataclass
class TournamentResearchReport:
    """Full research output for all matchups in a tournament."""
    tournament_name: str
    event_id: int
    report_date: str
    matchup_briefings: List[MatchupBriefing] = field(default_factory=list)
    total_players_researched: int = 0
    total_articles_found: int = 0


# ==============================================================================
# CLAUDE API PROMPTS
# ==============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are a golf research analyst providing factual intelligence briefings about PGA Tour players. Your role is STRICTLY informational — you do NOT make betting recommendations or predictions.

You analyze recent news articles and produce structured reports highlighting factors that could affect a player's upcoming tournament performance. Be concise, specific, and always cite sources. If no relevant information exists for a category, say "No reports found."

Always respond with valid JSON matching the requested schema."""

ANALYSIS_USER_PROMPT = """Analyze the following recent news about **{player_name}** ahead of the **{tournament_name}**.

{articles_text}

Respond with this exact JSON structure:
{{
    "injury_status": "Any injury, illness, or physical concern. Include dates and severity if known. Say 'No reports found' if none.",
    "equipment_changes": "Any club, ball, putter, or caddie changes. Include what changed and when. Say 'No reports found' if none.",
    "recent_form_commentary": "What media/analysts say about recent performance. Include specific tournament results mentioned (finishes, scores, streaks).",
    "personal_events": "Significant off-course events (family, legal, sponsorship, team changes). Only include if potentially relevant to performance. Say 'No reports found' if none.",
    "course_history_comments": "Any mentions of the player's history at this specific course or similar course types. Say 'No reports found' if none.",
    "sentiment": "positive OR neutral OR negative — overall media sentiment about this player's form and prospects this week",
    "key_quotes": ["Up to 3 notable direct quotes with attribution, e.g. 'I feel great about my game right now' — Player Name (Golf Channel, Mar 8)"]
}}"""


# ==============================================================================
# NEWS RESEARCH AGENT
# ==============================================================================

class NewsResearchAgent:
    """
    Read-only intelligence layer that researches players in identified bets.

    NEVER modifies bets, probabilities, or decisions. Produces reports
    for human review only.

    Parameters
    ----------
    settings : Settings, optional
        Project configuration (provides API keys, cache dir).
        If None, imports and creates default Settings.
    cache_ttl_days : int, optional
        How many days to cache research per player before refreshing.
        Overrides settings.NEWS_CACHE_TTL_DAYS if provided.
    max_articles_per_player : int, optional
        Maximum articles to fetch per player.
        Overrides settings.NEWS_MAX_ARTICLES_PER_PLAYER if provided.
    """

    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
    RATE_LIMIT_DELAY: float = 0.5  # seconds between HTTP requests

    def __init__(
        self,
        settings=None,
        cache_ttl_days: Optional[int] = None,
        max_articles_per_player: Optional[int] = None,
    ):
        if settings is None:
            from config.settings import Settings
            settings = Settings()

        self.settings = settings
        self.cache_ttl_days = cache_ttl_days or settings.NEWS_CACHE_TTL_DAYS
        self.max_articles = max_articles_per_player or settings.NEWS_MAX_ARTICLES_PER_PLAYER
        self.search_days_back = settings.NEWS_SEARCH_DAYS_BACK
        self.claude_model = settings.NEWS_CLAUDE_MODEL

        # Cache directory
        self.cache_dir = settings.PROJECT_ROOT / "cache" / "research"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HTTP session for news fetching
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })

        # Rate limiting
        self._last_request_time: float = 0.0

        # Anthropic client (lazy init)
        self._anthropic_client = None

        logger.info(
            "NewsResearchAgent initialized | cache_ttl=%dd | max_articles=%d | model=%s",
            self.cache_ttl_days, self.max_articles, self.claude_model,
        )

    @property
    def anthropic_client(self):
        """Lazy-initialize Anthropic client."""
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.settings.ANTHROPIC_API_KEY,
            )
        return self._anthropic_client

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def research_bets(
        self,
        bets_df: pd.DataFrame,
        tournament_name: str,
        event_id: int,
    ) -> TournamentResearchReport:
        """
        Research all players in the bets DataFrame.

        Parameters
        ----------
        bets_df : pd.DataFrame
            Identified bets with columns: matchup, bet_on, bet_on_id,
            edge, odds, stake (at minimum).
        tournament_name : str
            Name of the target tournament.
        event_id : int
            DataGolf event ID.

        Returns
        -------
        TournamentResearchReport
        """
        report = TournamentResearchReport(
            tournament_name=tournament_name,
            event_id=event_id,
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

        if bets_df is None or len(bets_df) == 0:
            logger.info("No bets to research.")
            return report

        # Extract unique players from matchups
        # matchup column format: "Player A vs Player B"
        player_cache: Dict[str, PlayerBriefing] = {}
        matchup_briefings = []

        for _, row in bets_df.iterrows():
            matchup_str = row.get("matchup", "")
            parts = matchup_str.split(" vs ")
            if len(parts) != 2:
                logger.warning("Cannot parse matchup: %s", matchup_str)
                continue

            player_a_name = parts[0].strip()
            player_b_name = parts[1].strip()
            bet_on = row.get("bet_on", player_a_name)

            # Determine player IDs (use bet_on_id for the bet-on player)
            bet_on_id = int(row.get("bet_on_id", 0))
            if bet_on == player_a_name:
                player_a_id = bet_on_id
                player_b_id = 0  # unknown, but name is enough for research
            else:
                player_a_id = 0
                player_b_id = bet_on_id

            # Research each player (deduped via cache)
            if player_a_name not in player_cache:
                player_cache[player_a_name] = self._research_player(
                    player_a_name, player_a_id, tournament_name,
                )
            if player_b_name not in player_cache:
                player_cache[player_b_name] = self._research_player(
                    player_b_name, player_b_id, tournament_name,
                )

            matchup_briefings.append(MatchupBriefing(
                matchup=matchup_str,
                bet_on=bet_on,
                edge=float(row.get("edge", 0)),
                stake=float(row.get("stake", 0)),
                odds=float(row.get("odds", 0)),
                player_a=player_cache[player_a_name],
                player_b=player_cache[player_b_name],
            ))

        report.matchup_briefings = matchup_briefings
        report.total_players_researched = len(player_cache)
        report.total_articles_found = sum(
            b.articles_found for b in player_cache.values()
        )

        logger.info(
            "Research complete | %d matchups | %d players | %d articles",
            len(matchup_briefings),
            report.total_players_researched,
            report.total_articles_found,
        )

        return report

    # ==========================================================================
    # PLAYER RESEARCH
    # ==========================================================================

    def _research_player(
        self, player_name: str, player_id: int, tournament_name: str,
    ) -> PlayerBriefing:
        """Fetch news + analyze for a single player. Uses cache if fresh."""
        # Check cache first
        cached = self._load_from_cache(player_name)
        if cached is not None:
            logger.info("Cache hit for %s", player_name)
            return cached

        logger.info("Researching %s ...", player_name)

        # Fetch news articles
        articles = self._fetch_google_news_rss(player_name)

        if not articles:
            briefing = PlayerBriefing(
                player_name=player_name,
                player_id=player_id,
                articles_found=0,
                confidence="none",
            )
            self._save_to_cache(player_name, briefing)
            return briefing

        # Fetch full text for top articles
        for article in articles[:self.max_articles]:
            if not article.snippet:
                text = self._fetch_article_text(article.url)
                if text:
                    article.snippet = text

        # Analyze with Claude
        analysis = self._analyze_with_claude(player_name, articles, tournament_name)

        # Determine confidence based on article count
        n = len(articles)
        if n >= 5:
            confidence = "high"
        elif n >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        # Build briefing
        sentiment_str = analysis.get("sentiment", "unknown").lower()
        try:
            sentiment = Sentiment(sentiment_str)
        except ValueError:
            sentiment = Sentiment.UNKNOWN

        briefing = PlayerBriefing(
            player_name=player_name,
            player_id=player_id,
            articles_found=n,
            injury_status=analysis.get("injury_status", "No reports found"),
            equipment_changes=analysis.get("equipment_changes", "No reports found"),
            recent_form_commentary=analysis.get("recent_form_commentary", "No reports found"),
            personal_events=analysis.get("personal_events", "No reports found"),
            course_history_comments=analysis.get("course_history_comments", "No reports found"),
            sentiment=sentiment,
            key_quotes=analysis.get("key_quotes", []),
            confidence=confidence,
            raw_articles=articles[:self.max_articles],
        )

        self._save_to_cache(player_name, briefing)
        return briefing

    # ==========================================================================
    # NEWS FETCHING
    # ==========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _fetch_google_news_rss(self, player_name: str) -> List[NewsArticle]:
        """Fetch recent golf articles via Google News RSS."""
        self._rate_limit()

        query = f"{player_name} golf"
        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        url = f"{self.GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"

        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Google News RSS failed for %s: %s", player_name, e)
            return []

        # Parse RSS XML
        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError as e:
            logger.warning("RSS XML parse error for %s: %s", player_name, e)
            return []

        articles = []
        cutoff = datetime.now() - timedelta(days=self.search_days_back)

        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pub_date_el = item.find("pubDate")
            source_el = item.find("source")
            desc_el = item.find("description")

            if title_el is None or link_el is None:
                continue

            # Parse publication date and filter by recency
            pub_date_str = pub_date_el.text if pub_date_el is not None else None
            if pub_date_str:
                try:
                    # Google News RSS date format: "Tue, 04 Mar 2026 12:00:00 GMT"
                    pub_date = datetime.strptime(
                        pub_date_str.strip(), "%a, %d %b %Y %H:%M:%S %Z"
                    )
                    if pub_date < cutoff:
                        continue
                except ValueError:
                    pass  # keep article if date can't be parsed

            source_name = source_el.text if source_el is not None else "Unknown"
            snippet = ""
            if desc_el is not None and desc_el.text:
                # Strip HTML from description
                snippet = re.sub(r"<[^>]+>", "", desc_el.text).strip()

            articles.append(NewsArticle(
                title=title_el.text or "",
                source=source_name,
                url=link_el.text or "",
                published=pub_date_str,
                snippet=snippet,
            ))

        logger.info("Found %d articles for %s (last %d days)",
                     len(articles), player_name, self.search_days_back)

        return articles[:self.max_articles]

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _fetch_article_text(self, url: str, max_chars: int = 3000) -> str:
        """Fetch and extract text from an article URL."""
        self._rate_limit()

        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug("Article fetch failed: %s — %s", url, e)
            return ""

        # Extract text — try BeautifulSoup first, fallback to regex
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Get text from paragraphs (most reliable for articles)
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
        except ImportError:
            # Fallback: strip HTML tags with regex
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars] if text else ""

    # ==========================================================================
    # CLAUDE ANALYSIS
    # ==========================================================================

    def _analyze_with_claude(
        self,
        player_name: str,
        articles: List[NewsArticle],
        tournament_name: str,
    ) -> dict:
        """
        Send articles to Claude API for structured analysis.

        Returns dict matching PlayerBriefing fields.
        Falls back to empty dict on failure.
        """
        # Build articles text block
        articles_text_parts = []
        for i, article in enumerate(articles[:self.max_articles], 1):
            parts = [f"**Article {i}:** {article.title}"]
            parts.append(f"Source: {article.source}")
            if article.published:
                parts.append(f"Date: {article.published}")
            if article.snippet:
                parts.append(f"Content: {article.snippet[:1500]}")
            articles_text_parts.append("\n".join(parts))

        articles_text = "\n\n---\n\n".join(articles_text_parts)

        if not articles_text.strip():
            return {}

        user_prompt = ANALYSIS_USER_PROMPT.format(
            player_name=player_name,
            tournament_name=tournament_name,
            articles_text=articles_text,
        )

        try:
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=1024,
                temperature=0.0,
                system=ANALYSIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse JSON from response
            response_text = response.content[0].text.strip()

            # Handle potential markdown code fences
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)

            result = json.loads(response_text)
            logger.info("Claude analysis complete for %s", player_name)
            return result

        except json.JSONDecodeError as e:
            logger.warning("Claude returned invalid JSON for %s: %s", player_name, e)
            return {}
        except Exception as e:
            logger.warning("Claude API error for %s: %s", player_name, e)
            return {}

    # ==========================================================================
    # CACHING
    # ==========================================================================

    def _cache_key(self, player_name: str) -> str:
        """Cache key: player name + ISO week number."""
        iso_week = datetime.now().strftime("%Y-W%V")
        raw = f"{player_name.lower().strip()}|{iso_week}"
        return md5(raw.encode()).hexdigest()

    def _cache_path(self, player_name: str) -> Path:
        return self.cache_dir / f"{self._cache_key(player_name)}.json"

    def _load_from_cache(self, player_name: str) -> Optional[PlayerBriefing]:
        """Load cached briefing if it exists and is within TTL."""
        path = self._cache_path(player_name)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            # Check TTL
            cached_date = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_date > timedelta(days=self.cache_ttl_days):
                logger.debug("Cache expired for %s", player_name)
                return None
            return PlayerBriefing.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("Cache read error for %s: %s", player_name, e)
            return None

    def _save_to_cache(self, player_name: str, briefing: PlayerBriefing):
        """Save briefing to disk cache."""
        path = self._cache_path(player_name)
        data = briefing.to_dict()
        data["_cached_at"] = datetime.now().isoformat()
        try:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            logger.debug("Cached research for %s", player_name)
        except OSError as e:
            logger.warning("Cache write failed for %s: %s", player_name, e)

    # ==========================================================================
    # RATE LIMITING
    # ==========================================================================

    def _rate_limit(self):
        """Enforce minimum delay between HTTP requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    # ==========================================================================
    # DISPLAY
    # ==========================================================================

    def display_report(self, report: TournamentResearchReport):
        """
        Render the research report for Jupyter notebook display.

        Uses IPython HTML if available, falls back to plain text.
        """
        if not report.matchup_briefings:
            print("No matchup research to display.")
            return

        try:
            from IPython.display import display, HTML
            html = self._render_html(report)
            display(HTML(html))
        except ImportError:
            self._print_plain(report)

    def _render_html(self, report: TournamentResearchReport) -> str:
        """Build HTML string for Jupyter display."""
        SENTIMENT_COLORS = {
            Sentiment.POSITIVE: "#28a745",
            Sentiment.NEUTRAL: "#6c757d",
            Sentiment.NEGATIVE: "#dc3545",
            Sentiment.UNKNOWN: "#adb5bd",
        }

        CONFIDENCE_BADGES = {
            "high": "&#9679;&#9679;&#9679;",
            "medium": "&#9679;&#9679;&#9675;",
            "low": "&#9679;&#9675;&#9675;",
            "none": "&#9675;&#9675;&#9675;",
        }

        parts = [
            '<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; '
            'max-width: 900px; margin: 0 auto;">',
            f'<h2 style="border-bottom: 3px solid #333; padding-bottom: 8px;">'
            f'Research Briefing: {report.tournament_name}</h2>',
            f'<p style="color: #666; margin-top: -8px;">'
            f'{report.report_date} &nbsp;|&nbsp; '
            f'{report.total_players_researched} players researched &nbsp;|&nbsp; '
            f'{report.total_articles_found} articles analyzed</p>',
        ]

        for i, mb in enumerate(report.matchup_briefings, 1):
            bet_badge = (
                f'<span style="background: #e8f5e9; color: #2e7d32; padding: 2px 8px; '
                f'border-radius: 4px; font-size: 0.85em;">'
                f'Bet: {mb.bet_on} @ {mb.odds:.2f} | '
                f'Edge: {mb.edge:.1%} | Stake: ${mb.stake:.0f}</span>'
            )

            parts.append(
                f'<details style="margin: 16px 0; border: 1px solid #ddd; '
                f'border-radius: 8px; overflow: hidden;" open>'
                f'<summary style="background: #f8f9fa; padding: 12px 16px; '
                f'cursor: pointer; font-weight: 600; font-size: 1.05em;">'
                f'Matchup {i}: {mb.matchup} &nbsp;&nbsp;{bet_badge}</summary>'
                f'<div style="padding: 16px;">'
            )

            # Render both players
            for player_briefing in [mb.player_a, mb.player_b]:
                is_bet_on = player_briefing.player_name == mb.bet_on
                border_color = "#28a745" if is_bet_on else "#ddd"
                label = " (YOUR BET)" if is_bet_on else ""

                sent_color = SENTIMENT_COLORS.get(player_briefing.sentiment, "#adb5bd")
                conf_badge = CONFIDENCE_BADGES.get(player_briefing.confidence, "")

                parts.append(
                    f'<div style="border-left: 4px solid {border_color}; '
                    f'padding: 12px 16px; margin: 12px 0; background: #fafafa; '
                    f'border-radius: 0 6px 6px 0;">'
                    f'<h4 style="margin: 0 0 8px 0;">'
                    f'{player_briefing.player_name}{label} &nbsp;'
                    f'<span style="color: {sent_color}; font-size: 0.85em;">'
                    f'[{player_briefing.sentiment.value.upper()}]</span> &nbsp;'
                    f'<span style="font-size: 0.75em; color: #999;" '
                    f'title="Research confidence">{conf_badge}</span>'
                    f'</h4>'
                )

                fields = [
                    ("Injury", player_briefing.injury_status),
                    ("Form", player_briefing.recent_form_commentary),
                    ("Equipment", player_briefing.equipment_changes),
                    ("Personal", player_briefing.personal_events),
                    ("Course", player_briefing.course_history_comments),
                ]

                for label_text, value in fields:
                    if value and value != "No reports found":
                        parts.append(
                            f'<p style="margin: 4px 0; font-size: 0.9em;">'
                            f'<strong>{label_text}:</strong> {value}</p>'
                        )
                    else:
                        parts.append(
                            f'<p style="margin: 4px 0; font-size: 0.9em; color: #999;">'
                            f'<strong>{label_text}:</strong> {value}</p>'
                        )

                # Key quotes
                if player_briefing.key_quotes:
                    parts.append(
                        '<div style="margin-top: 8px; padding-left: 12px; '
                        'border-left: 2px solid #ddd;">'
                    )
                    for quote in player_briefing.key_quotes[:3]:
                        parts.append(
                            f'<p style="margin: 4px 0; font-size: 0.85em; '
                            f'font-style: italic; color: #555;">"{quote}"</p>'
                        )
                    parts.append('</div>')

                parts.append('</div>')  # player div

            parts.append('</div></details>')  # matchup details

        parts.append('</div>')  # root div
        return "\n".join(parts)

    def _print_plain(self, report: TournamentResearchReport):
        """Plain text fallback for non-Jupyter environments."""
        print(f"\n{'='*70}")
        print(f"RESEARCH BRIEFING: {report.tournament_name}")
        print(f"Date: {report.report_date} | "
              f"Players: {report.total_players_researched} | "
              f"Articles: {report.total_articles_found}")
        print(f"{'='*70}")

        for i, mb in enumerate(report.matchup_briefings, 1):
            print(f"\n--- Matchup {i}: {mb.matchup} ---")
            print(f"[Bet: {mb.bet_on} @ {mb.odds:.2f} | "
                  f"Edge: {mb.edge:.1%} | Stake: ${mb.stake:.0f}]")

            for player_briefing in [mb.player_a, mb.player_b]:
                is_bet_on = player_briefing.player_name == mb.bet_on
                marker = " << YOUR BET" if is_bet_on else ""
                print(f"\n  {player_briefing.player_name}"
                      f" [{player_briefing.sentiment.value.upper()}]{marker}")
                print(f"    Injury:    {player_briefing.injury_status}")
                print(f"    Form:      {player_briefing.recent_form_commentary}")
                print(f"    Equipment: {player_briefing.equipment_changes}")
                print(f"    Personal:  {player_briefing.personal_events}")
                print(f"    Course:    {player_briefing.course_history_comments}")

                if player_briefing.key_quotes:
                    for quote in player_briefing.key_quotes[:3]:
                        print(f'    > "{quote}"')

        print(f"\n{'='*70}\n")
