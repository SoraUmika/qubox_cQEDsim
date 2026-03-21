param(
    [string]$Task,
    [switch]$Resume,
    [switch]$ResumeLast,
    [switch]$ForceRestart,
    [int]$MaxIterations = 0,
    [switch]$DryRun,
    [switch]$Verbose
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @(
    "E:\Program Files\Python311\python.exe",
    "E:\Programs\python.exe",
    "C:\Users\dazzl\AppData\Local\Programs\Python\Python312\python.exe"
)
$python = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        $python = $command.Source
    }
}
if (-not $python) {
    throw "Could not find a usable Python interpreter."
}

$arguments = @("$repoRoot\tools\run_agent_workflow.py")
if ($Task) {
    $arguments += @("--task", $Task)
}
if ($Resume) {
    $arguments += "--resume"
}
if ($ResumeLast) {
    $arguments += "--resume-last"
}
if ($ForceRestart) {
    $arguments += "--force-restart"
}
if ($MaxIterations -gt 0) {
    $arguments += @("--max-iterations", $MaxIterations.ToString())
}
if ($DryRun) {
    $arguments += "--dry-run"
}
if ($Verbose) {
    $arguments += "--verbose"
}

& $python @arguments
exit $LASTEXITCODE
