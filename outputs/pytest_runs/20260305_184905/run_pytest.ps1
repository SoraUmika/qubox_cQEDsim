$ErrorActionPreference = 'Stop'
$start = Get-Date
$log = 'outputs\pytest_runs\20260305_184905\pytest_full.log'
$exitFile = 'outputs\pytest_runs\20260305_184905\exit_code.txt'
$summaryFile = 'outputs\pytest_runs\20260305_184905\summary.txt'
python -m pytest -q *>&1 | Tee-Object -FilePath $log
$rc = $LASTEXITCODE
$end = Get-Date
@(
  "start_utc=$(($start).ToUniversalTime().ToString('o'))",
  "end_utc=$(($end).ToUniversalTime().ToString('o'))",
  "duration_s=$([math]::Round(($end-$start).TotalSeconds,3))",
  "exit_code=$rc"
) | Out-File -FilePath $summaryFile -Encoding ascii
"$rc" | Out-File -FilePath $exitFile -Encoding ascii
exit $rc
