$ErrorActionPreference = 'Stop'
$env:GIT_CONFIG_COUNT = '0'
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'Never'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path.Replace('\','/')
$credentialText = "protocol=https`nhost=github.com`nusername=ahesh-dilhan`n`n" | git credential fill
if ($LASTEXITCODE -ne 0) { throw 'GitHub credential lookup failed.' }
$credential = @{}
foreach ($line in $credentialText) {
    if ($line -match '^([^=]+)=(.*)$') { $credential[$Matches[1]] = $Matches[2] }
}
if (-not $credential.password) { throw 'No cached GitHub authentication available.' }
$headers = @{ Authorization = 'Bearer ' + $credential.password; Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2022-11-28' }
$account = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers
if ($account.login -ne 'ahesh-dilhan') { throw 'Unexpected authenticated GitHub account.' }
$repo = $null
try { $repo = Invoke-RestMethod -Uri 'https://api.github.com/repos/ahesh-dilhan/en3160-assignment-03' -Headers $headers }
catch { if ($_.Exception.Response.StatusCode.value__ -ne 404) { throw } }
if ($repo) { throw 'Repository already exists; inspect it before choosing how to publish.' }
$body = @{ name = 'en3160-assignment-03'; description = 'EN3160 Assignment 3: manual neural-network gradients, MNIST CNN comparison and ResNet18 transfer learning. 230145E.'; private = $false; auto_init = $false } | ConvertTo-Json
$repo = Invoke-RestMethod -Method Post -Uri 'https://api.github.com/user/repos' -Headers $headers -ContentType 'application/json' -Body $body
if ($repo.private) { throw 'Incorrect repository visibility.' }
git -c "safe.directory=$repositoryRoot" remote add origin $repo.clone_url
if ($LASTEXITCODE -ne 0) { throw 'Could not add remote.' }
git -c "safe.directory=$repositoryRoot" push -u origin main
if ($LASTEXITCODE -ne 0) { throw 'Could not push commit.' }
Write-Output ('Created public repository: ' + $repo.html_url)
