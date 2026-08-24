/**
 * EXP-FX000016/SYS-FX022 Stage 1・コンポーネントA: ライブ気配値の記録.
 *
 * GitHub Actionsが無料枠超過で使えず、Claude Code Remoteの使い捨てセッション
 * Routineはgit pushが完了しない不具合があったため、既存のFirebase/GCPプロ
 * ジェクトのCloud Scheduler(毎時)+Cloud Functionsで代替する。
 *
 * GMO公開Ticker API(認証不要)を1回呼び出し、SYS-FX012対象4通貨のbid/ask/
 * 実測スプレッドを取得し、GitHub Contents APIで直接
 * data/raw/live-ticker/YYYY-MM.csv へ追記コミットする(git clone不要、
 * サーバーレス環境に向いたAPI経由の更新方式)。
 *
 * 実発注は一切行わない。GMO側の認証情報も使わない(公開エンドポイントのみ)。
 *
 * 必要なSecret: GITHUB_TOKEN
 *   - GitHub側で当リポジトリのみに限定したFine-grained PAT を発行し
 *     (権限は Contents: Read and write のみ)、
 *     `firebase functions:secrets:set GITHUB_TOKEN` で登録する。
 *     トークンの値は本コードにもチャットにも含めない。
 */

const {onSchedule} = require("firebase-functions/v2/scheduler");
const {defineSecret} = require("firebase-functions/params");
const {logger} = require("firebase-functions");

const GITHUB_TOKEN = defineSecret("GITHUB_TOKEN");

const REPO_OWNER = "hasebe201114-cyber";
const REPO_NAME = "minmax-fx-day-trading-lab";
const BRANCH = "main";
const PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]; // SYS-FX012凍結設計と同一
const TICKER_URL = "https://forex-api.coin.z.com/public/v1/ticker";
const CSV_HEADER = "polled_at,pair,bid,ask,spread_pips,api_timestamp,market_status\n";

function pipSize(pair) {
  return pair.endsWith("JPY") ? 0.01 : 0.0001;
}

async function fetchTickerRows() {
  const resp = await fetch(TICKER_URL);
  if (!resp.ok) {
    throw new Error(`GMO ticker API error: ${resp.status}`);
  }
  const json = await resp.json();
  const byPair = Object.fromEntries((json.data || []).map((r) => [r.symbol, r]));
  const polledAt = new Date().toISOString();

  const rows = [];
  for (const pair of PAIRS) {
    const row = byPair[pair];
    if (!row) continue;
    const bid = parseFloat(row.bid);
    const ask = parseFloat(row.ask);
    const spreadPips = (ask - bid) / pipSize(pair);
    rows.push([polledAt, pair, bid, ask, spreadPips.toFixed(4), row.timestamp, row.status].join(","));
  }
  return {polledAt, rows};
}

async function appendToGithubCsv(token, path, newLines, commitMessage) {
  const apiBase = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/contents/${path}`;
  const headers = {
    "Authorization": `Bearer ${token}`,
    "Accept": "application/vnd.github+json",
    "User-Agent": "sysfx022-live-ticker-poll",
  };

  let sha;
  let existingContent = CSV_HEADER;
  const getResp = await fetch(`${apiBase}?ref=${BRANCH}`, {headers});
  if (getResp.status === 200) {
    const data = await getResp.json();
    sha = data.sha;
    existingContent = Buffer.from(data.content, "base64").toString("utf-8");
    if (!existingContent.endsWith("\n")) existingContent += "\n";
  } else if (getResp.status !== 404) {
    throw new Error(`GitHub GET failed: ${getResp.status} ${await getResp.text()}`);
  }

  const newContent = existingContent + newLines.join("\n") + "\n";
  const putResp = await fetch(apiBase, {
    method: "PUT",
    headers: {...headers, "Content-Type": "application/json"},
    body: JSON.stringify({
      message: commitMessage,
      content: Buffer.from(newContent, "utf-8").toString("base64"),
      sha,
      branch: BRANCH,
    }),
  });
  if (!putResp.ok) {
    throw new Error(`GitHub PUT failed: ${putResp.status} ${await putResp.text()}`);
  }
}

exports.pollLiveTicker = onSchedule(
    {schedule: "5 * * * *", timeZone: "UTC", secrets: [GITHUB_TOKEN], retryCount: 0},
    async () => {
      const {polledAt, rows} = await fetchTickerRows();
      if (rows.length === 0) {
        logger.warn("取得できたレコードが0件でした(API応答に対象通貨が含まれない)");
        return;
      }
      const monthTag = polledAt.slice(0, 7); // YYYY-MM
      const path = `data/raw/live-ticker/${monthTag}.csv`;
      await appendToGithubCsv(
          GITHUB_TOKEN.value(), path, rows, `ライブ気配値記録: ${polledAt}`,
      );
      logger.info(`${rows.length}件を${path}へ追記コミットしました`);
    },
);
