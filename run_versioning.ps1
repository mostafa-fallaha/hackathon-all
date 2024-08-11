param (
    [string]$pythonFile,
    [string]$commitMessage
)

if (-not $pythonFile -or -not $commitMessage) {
    Write-Host "Usage: .\run_versioning.ps1 <python_file> <commit_message>"
    exit 1
}

if (-not (Test-Path $pythonFile)) {
    Write-Host "Error: The specified Python file '$pythonFile' does not exist."
    exit 1
}

try {
    # Run the specified Python script with the commit message
    python $pythonFile $commitMessage
} catch {
    Write-Host "An error occurred: $($_.Exception.Message)"
    exit 1
}
