#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ruleName = "TabbyAPI OAI (TCP 5000 LAN)"
$existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

if ($null -eq $existingRule) {
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 5000 `
        -Profile Private `
        -RemoteAddress LocalSubnet | Out-Null
} else {
    $existingRule | Set-NetFirewallRule `
        -Enabled True `
        -Profile Private `
        -Direction Inbound `
        -Action Allow `
        -RemoteAddress LocalSubnet
}

Write-Host "TabbyAPI TCP 5000 is allowed for the local subnet on Private networks."
