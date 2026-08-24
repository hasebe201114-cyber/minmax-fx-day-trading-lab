# EXP-FX000016/SYS-FX022 Stage 1・コンポーネントA: Firebase Cloud Functions版

GitHub Actionsが無料枠超過、Claude Code Remoteの使い捨てセッションRoutineは
git pushが完了しない不具合があったため、既存のFirebase/GCPプロジェクトの
Cloud Scheduler + Cloud Functions(2nd gen)で代替する。

`index.js`の`pollLiveTicker`が毎時(UTC 5分)起動し、GMO公開Ticker API(認証
不要)を1回呼び出して`data/raw/live-ticker/YYYY-MM.csv`へ、GitHub Contents
API経由で直接追記コミットする。**このセッションにはFirebase/GCPのCLI・認証
情報が無いため、以下は司令塔側での実施が必要。**

## 事前準備(1回だけ)

1. GitHubで、このリポジトリ専用のFine-grained Personal Access Tokenを発行する
   - GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
   - Repository access: `hasebe201114-cyber/minmax-fx-day-trading-lab` のみに限定
   - Permissions: **Contents: Read and write** のみ(他の権限は付与しない)
   - トークンの値は司令塔の手元にのみ保管し、Claudeとの会話には貼り付けないこと

2. 既存のFirebase/GCPプロジェクトでこの関数用のSecretを設定する:
   ```bash
   firebase use <既存のプロジェクトID>
   firebase functions:secrets:set GITHUB_TOKEN
   # プロンプトで上記トークンの値を入力(標準入力、画面に表示されない)
   ```

## デプロイ

```bash
cd functions
npm install
cd ..
firebase deploy --only functions:pollLiveTicker
```

Cloud Scheduler側のジョブはFirebase Functionsのデプロイ時に自動作成される
(2nd gen `onSchedule`の仕様)。GCPコンソールの Cloud Scheduler 画面で
`pollLiveTicker`ジョブが毎時作成されていることを確認できる。

## 動作確認

デプロイ後、GCPコンソールまたは以下で手動トリガーして確認:
```bash
gcloud scheduler jobs run pollLiveTicker --location=<region>
```

成功していれば数十秒後に`data/raw/live-ticker/YYYY-MM.csv`へ新しいコミット
が`main`に反映される。Cloud Functionsのログ(Cloud Logging)でもエラーの
有無を確認できる。

## 費用の目安

毎時1回・1回あたりGMO APIとGitHub APIへそれぞれ1〜2リクエストのみの軽量な
関数。Cloud Functions無料枠(月200万回呼び出し)に対して月720回程度なので、
通常のFirebase/GCP利用状況であれば追加費用はほぼ発生しない見込み(既存
プロジェクトの無料枠残量による)。

## 停止・削除する場合

```bash
firebase functions:delete pollLiveTicker
```
