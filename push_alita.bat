@echo off
REM Batch script to push changes to the Alita GitHub repository on the develop branch

REM Navigate to the Alita repository directory
cd C:\Alita

REM Add all changes to staging
git add .

REM Commit the changes with a message
git commit -m "Your commit message for Alita"

REM Push the changes to the develop branch
git push origin develop

REM Optional: Pause the script to keep the window open
pause
