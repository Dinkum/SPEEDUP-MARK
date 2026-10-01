"""Grade current candidates, or manage an agent run with the run subcommand."""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "run":
    from .run_manager import main

    main(sys.argv[2:])
else:
    from .harness import main

    main()
