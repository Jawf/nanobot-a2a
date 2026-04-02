---
name: ai-news
description: Fetch and summarize the latest AI news, research papers, and industry updates. Triggered by phrases like "AI news", "AI日报", "today's AI updates", "最新AI资讯", or when run on schedule.
metadata: {"nanobot":{"emoji":"🤖"}}
---

# AI News Daily Digest

Compile a concise daily digest of the most important AI news from the past 24 hours.

## Data Sources (in priority order)

### 1. HackerNews AI Posts
```
https://hn.algolia.com/api/v1/search?tags=story&query=AI&numericFilters=created_at_i>TIMESTAMP&hitsPerPage=20
```
Replace `TIMESTAMP` with Unix timestamp for 24 hours ago. Pick posts with `points >= 50`.

### 2. ArXiv — Latest Papers
Fetch the RSS feed for new submissions:
```
https://rss.arxiv.org/rss/cs.AI
https://rss.arxiv.org/rss/cs.LG
```
Pick the top 3–5 papers by relevance/novelty.

### 3. Key AI News Sites (via web search)
Search for today's top stories:
- Query: `AI news today site:venturebeat.com OR site:techcrunch.com OR site:theverge.com OR site:arstechnica.com`
- Query: `大模型 AI 新闻 最新` (for Chinese-language coverage)

### 4. Official Release Announcements
Search: `"announced" OR "released" OR "launched" AI model 2026`

## Digest Format

Structure the output as follows (adapt language to match user's language):

```
🤖 AI 日报 · {DATE}
━━━━━━━━━━━━━━━━━━━━

🔬 研究前沿
• [paper title] — [one-line summary] (ArXiv)
• ...

🚀 产品动态
• [product/company] — [what happened] 
• ...

💡 行业观察
• [trend or insight]
• ...

🔗 热门讨论 (HN)
• [post title] ★{points} — [link]
• ...
```

Keep each item to 1–2 lines. Total digest should be under 600 words.

## Execution Steps

1. Compute the Unix timestamp for 24 hours ago.
2. Fetch HackerNews via the Algolia API (HTTP GET, no auth needed).
3. Fetch ArXiv RSS feeds and parse titles + abstracts.
4. Run 1–2 web searches for industry news.
5. Deduplicate and rank by significance.
6. Write the digest in the user's preferred language (default: Chinese).
7. If triggered via cron with `deliver: true`, send via the configured channel.

## Notes

- If a fetch fails, skip that source silently and proceed with others.
- Avoid repeating items that appeared in previous digests (check HISTORY.md if available).
- For cron delivery, keep the tone friendly and scannable — this is a morning briefing.
