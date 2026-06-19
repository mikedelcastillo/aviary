# Launch the custom bird annotation web tool on http://0.0.0.0:5000.
# Filesystem is the database: reads/writes annotations next to the raw images
# under data/annotation/raw, labels sourced from training/roster.yaml.
# Equivalent to scripts/annotation.sh on Windows.
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

# Absolute paths so the Next.js server resolves them regardless of cwd.
$env:AVIARY_DATA_ROOT = Join-Path $RepoRoot "data/annotation/raw"
$env:AVIARY_ROSTER = Join-Path $RepoRoot "training/roster.yaml"

# Raise the libuv worker-thread pool so the many concurrent file reads/stats and
# image decodes during the dedupe hashing pass don't queue behind the default 4
# threads (Windows has no `ulimit -n`; this is the equivalent "more concurrent
# file ops" lever). Must be set before the Node process starts.
$env:UV_THREADPOOL_SIZE = "16"

# Ensure other devices on the LAN can reach the server. The Next.js server binds
# to 0.0.0.0, but Windows Firewall blocks inbound TCP 5000 by default, so without
# this rule http://<lan-ip>:5000 only works from this machine.
$FirewallRuleName = "Aviary annotation 5000"
$existingRule = Get-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue
if (-not $existingRule) {
    $isAdmin = ([Security.Principal.WindowsPrincipal] `
        [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if ($isAdmin) {
        Write-Host "Adding firewall rule '$FirewallRuleName' (inbound TCP 5000)..."
        New-NetFirewallRule -DisplayName $FirewallRuleName -Direction Inbound `
            -Action Allow -Protocol TCP -LocalPort 5000 -Profile Private | Out-Null
    }
    else {
        Write-Host "Firewall rule '$FirewallRuleName' is missing; requesting elevation to add it..."
        $addRule = "New-NetFirewallRule -DisplayName '$FirewallRuleName' -Direction Inbound " +
            "-Action Allow -Protocol TCP -LocalPort 5000 -Profile Private"
        try {
            Start-Process powershell.exe -Verb RunAs -Wait -ArgumentList @(
                "-NoProfile", "-Command", $addRule
            )
            Write-Host "Firewall rule added."
        }
        catch {
            Write-Warning ("Could not add firewall rule (elevation declined). " +
                "LAN access from other devices will be blocked. Run this script as " +
                "Administrator, or add the rule manually:`n  $addRule")
        }
    }
}
else {
    Write-Host "Firewall rule '$FirewallRuleName' already present."
}

Push-Location (Join-Path $RepoRoot "annotation")
try {
    if (-not (Test-Path node_modules)) {
        Write-Host "Installing annotation tool dependencies (first run)..."
        npm install
    }

    Write-Host "Building annotation tool..."
    npm run build

    Write-Host "Annotation tool -> http://0.0.0.0:5000"
    npm run start
}
finally {
    Pop-Location
}
