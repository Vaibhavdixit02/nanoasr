import sys

if len(sys.argv) > 1 and sys.argv[1] == "transcribe":
    sys.argv.pop(1)
    from nanoasr.transcribe import main
else:
    from nanoasr.train import main

main()
