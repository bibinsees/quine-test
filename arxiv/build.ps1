# Local compile check (not uploaded to arXiv). Builds in a scratch directory so
# arxiv/ holds only the upload set: main.tex + figures.
$src = Split-Path -Parent $MyInvocation.MyCommand.Path
$b = Join-Path $env:TEMP "arxiv_build"
New-Item -ItemType Directory -Force $b | Out-Null
Copy-Item (Join-Path $src "*.tex") $b -Force
Copy-Item (Join-Path $src "*.png") $b -Force
Set-Location $b
$pdf = "C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
& $pdf -interaction=nonstopmode -halt-on-error main.tex > build1.txt
& $pdf -interaction=nonstopmode -halt-on-error main.tex > build2.txt
$errs = Select-String -Path main.log -Pattern '^! ' | Select-Object -First 10
if ($errs) { "ERRORS:"; $errs | ForEach-Object { $_.Line }; Select-String -Path main.log -Pattern '^l\.\d+' | Select-Object -First 5 | ForEach-Object { $_.Line } }
$over = Select-String -Path main.log -Pattern 'Overfull \\hbox \((\d+)' | Where-Object { [int]$_.Matches[0].Groups[1].Value -ge 15 }
"overfull boxes >= 15pt: $(@($over).Count)"
$over | Select-Object -First 6 | ForEach-Object { $_.Line }
$floats = Select-String -Path main.log -Pattern 'Float too large for page|Overfull \\vbox'
"oversized floats / vboxes: $(@($floats).Count)"
$floats | Select-Object -First 4 | ForEach-Object { $_.Line }
$pages = Select-String -Path main.log -Pattern 'Output written on main.pdf \((\d+) pages'
if ($pages) { "PDF: $($pages.Matches[0].Groups[1].Value) pages -> $b\main.pdf" } else { "NO PDF" }
Select-String -Path main.log -Pattern 'undefined references|Citation .* undefined|Reference .* undefined' | ForEach-Object { $_.Line }
