import sys

if len(sys.argv) > 1 and sys.argv[1] == "transcribe":
    sys.argv.pop(1)
    from nanoasr.transcribe import main
elif len(sys.argv) > 1 and sys.argv[1] == "live":
    sys.argv.pop(1)
    from nanoasr.live import main
else:
    from nanoasr.torch.train import main

main()
