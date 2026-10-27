$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.Encoding]::UTF8
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
Add-Type -AssemblyName System.Speech
$synth = [System.Speech.Synthesis.SpeechSynthesizer]::new()
$stream = [IO.MemoryStream]::new()
try {
    if ($request.voice) { $synth.SelectVoice($request.voice) }
    else {
        $english = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'en' })
        if (-not $english.Count) { throw 'No enabled English Windows voice is installed.' }
        $preferred = @($english | Where-Object { $_.VoiceInfo.Name -eq 'Microsoft David Desktop' })
        if ($preferred.Count) { $synth.SelectVoice($preferred[0].VoiceInfo.Name) }
        else { $synth.SelectVoice($english[0].VoiceInfo.Name) }
    }
    $synth.SetOutputToWaveStream($stream)
    # Speak plain text. Never interpret user input as PowerShell or SSML.
    $synth.Speak([string]$request.text)
    @{voice=$synth.Voice.Name; wav=[Convert]::ToBase64String($stream.ToArray())} | ConvertTo-Json -Compress
}
finally {
    $synth.Dispose()
    $stream.Dispose()
}
