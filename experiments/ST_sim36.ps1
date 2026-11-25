$ErrorActionPreference = 'Stop'
$jobs = [System.Collections.Generic.List[object]]::new()
$priority = @(@{Seed=2028;Obs=1}, @{Seed=2029;Obs=1}, @{Seed=2033;Obs=5}, @{Seed=2027;Obs=3}, @{Seed=2031;Obs=3})
foreach ($j in $priority) { $jobs.Add($j) }
$deferred = @(@{Seed=2032;Obs=3}, @{Seed=2032;Obs=5}, @{Seed=2036;Obs=5})
$deferredKeys = @($deferred | ForEach-Object { "$($_.Seed):$($_.Obs)" })
$priorityKeys = @($priority | ForEach-Object { "$($_.Seed):$($_.Obs)" })
foreach ($seed in 2027..2038) {
    foreach ($obs in @(1,3,5)) {
        $key = "$seed`:$obs"
        if ($priorityKeys -contains $key -or $deferredKeys -contains $key) { continue }
        $jobs.Add(@{Seed=$seed;Obs=$obs})
    }
}
foreach ($j in $deferred) { $jobs.Add($j) }

$outPath = Join-Path $PSScriptRoot 'ST_sim36_results.json'
$executionLog = Join-Path $PSScriptRoot 'ST_sim36_execution.log'
$all = [System.Collections.Generic.List[object]]::new()
$started = Get-Date
$deadline = (Get-Date).Date.AddHours(4).AddMinutes(45)
$physicalCount = 0
$invalidStreak = 0
$envBadStreak = 0
$setupRetries = 0
$firstInvalid = [System.Collections.Generic.List[bool]]::new()
$firstRates = [System.Collections.Generic.List[object]]::new()
$stopReason = 'completed_all_36'
$stop = $false

function Save-State {
    $obj = [ordered]@{
        started_at = $started.ToString('o')
        updated_at = (Get-Date).ToString('o')
        elapsed_hours = [math]::Round(((Get-Date)-$started).TotalHours,4)
        no_new_start_at = $deadline.ToString('o')
        planned_settings = 36
        completed_unique_settings = @($all | ForEach-Object { "$($_.seed):$($_.obstacles)" } | Select-Object -Unique).Count
        physical_trials = $physicalCount
        setup_retries = $setupRetries
        stop_reason = $stopReason
        results = @($all)
    }
    $json = ConvertTo-Json -InputObject $obj -Depth 50
    [System.IO.File]::WriteAllText($outPath,$json,[System.Text.UTF8Encoding]::new($false))
}

foreach ($job in $jobs) {
    if ($stop) { break }
    $seed=[int]$job.Seed; $obs=[int]$job.Obs; $retry=0
    do {
        if ((Get-Date) -ge $deadline) { $stopReason='no_new_trial_after_0445'; $stop=$true; break }
        if (((Get-Date)-$started).TotalHours -ge 5) { $stopReason='five_hour_limit'; $stop=$true; break }
        $cpuSamples=@(Get-CimInstance Win32_Processor | ForEach-Object { [double]$_.LoadPercentage })
        $hostLoad=if ($cpuSamples.Count) { [math]::Round(($cpuSamples | Measure-Object -Average).Average,1) } else { 'unavailable' }
        $tag="ST_${seed}_o${obs}_B"
        if ($retry -gt 0) { $tag += "_r$($retry+1)" }
        Write-Output "START $tag host_cpu=$hostLoad at=$((Get-Date).ToString('o'))"
        $raw=& docker exec -u misys crazy_shamir bash /home/misys/ST_one_run.sh $tag $seed $obs $hostLoad 2>&1
        $dockerExit=$LASTEXITCODE
        $resultLines=@($raw | ForEach-Object { [string]$_ } | Where-Object { $_ -like 'RESULT=*' } | Select-Object -Last 1)
        $physicalCount++
        if ($dockerExit -ne 0 -or $resultLines.Count -eq 0) {
            $result=[ordered]@{name=$tag;seed=$seed;obstacles=$obs;reason='host_or_container_no_response';invalid=$true;host_cpu_load_start=$hostLoad;docker_exit_code=$dockerExit;output=(@($raw)-join "`n")}
            $all.Add($result); $invalidStreak++; $firstInvalid.Add($true); Save-State
            $stopReason='host_or_container_no_response'; $stop=$true; break
        }
        try { $result=$resultLines[0].Substring(7) | ConvertFrom-Json }
        catch { $result=[pscustomobject]@{name=$tag;seed=$seed;obstacles=$obs;reason='unparseable_result';trace=$null;collision_at_t0=$false} }
        $traceN=0; $rate=$null
        if ($result.trace) { if ($null -ne $result.trace.n) { $traceN=[int]$result.trace.n }; if ($null -ne $result.trace.gt25_pct) { $rate=[double]$result.trace.gt25_pct } }
        $reason=[string]$result.reason
        $invalid=($reason -eq 'setup_error' -or [bool]$result.collision_at_t0 -or $reason -eq 'unparseable_result' -or $traceN -eq 0)
        $result | Add-Member -NotePropertyName invalid -NotePropertyValue $invalid -Force
        $result | Add-Member -NotePropertyName physical_trial_index -NotePropertyValue $physicalCount -Force
        $result | Add-Member -NotePropertyName priority_phase -NotePropertyValue ($jobs.IndexOf($job) -lt 5) -Force
        $result | Add-Member -NotePropertyName env_bad -NotePropertyValue $false -Force
        $all.Add($result); $firstInvalid.Add($invalid)
        if ($invalid) { $invalidStreak++ } else { $invalidStreak=0 }
        if ($physicalCount -le 2) { $firstRates.Add($rate) }
        if ($null -ne $rate -and $rate -gt 5) { $envBadStreak++ } else { $envBadStreak=0 }
        Save-State
        Write-Output "DONE $($result.name) reason=$reason invalid=$invalid gt25_pct=$rate consecutive_invalid=$invalidStreak consecutive_env_bad=$envBadStreak"

        if ($physicalCount -eq 2 -and $firstInvalid.Count -ge 2 -and $firstInvalid[0] -and $firstInvalid[1]) { $stopReason='first_two_trials_invalid'; $stop=$true; break }
        if ($physicalCount -eq 2 -and $firstRates.Count -ge 2 -and $null -ne $firstRates[0] -and $null -ne $firstRates[1] -and $firstRates[0] -gt 1 -and $firstRates[1] -gt 1) { $stopReason='first_two_gt25_over_1pct'; $stop=$true; break }
        if ($invalidStreak -ge 4) { $stopReason='four_consecutive_invalid'; $stop=$true; break }
        if ($envBadStreak -ge 3) {
            for ($k=1; $k -le 3; $k++) { $all[$all.Count-$k] | Add-Member -NotePropertyName env_bad -NotePropertyValue $true -Force }
            $stopReason='three_consecutive_gt25_over_5pct'; $stop=$true; Save-State; break
        }
        if ($invalid -and $reason -eq 'setup_error' -and $retry -eq 0 -and $setupRetries -lt 4 -and (Get-Date) -lt $deadline) {
            $retry=1; $setupRetries++; Write-Output "RETRY $tag as _r2 after setup_error"; continue
        }
        break
    } while ($true)
}
Save-State
Write-Output "STOP $stopReason unique=$(@($all | ForEach-Object { "$($_.seed):$($_.obstacles)" } | Select-Object -Unique).Count) physical=$physicalCount elapsed_hours=$([math]::Round(((Get-Date)-$started).TotalHours,4))"
