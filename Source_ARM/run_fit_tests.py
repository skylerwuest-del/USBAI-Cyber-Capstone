"""Run safe unit/regression tests and append results to the primary diagnostics log."""
import contextlib
import argparse
import io
from pathlib import Path
import sys
import unittest

from fit_diagnostics import DiagnosticRun


class CountingResult(unittest.TextTestResult):
    passed = 0

    def addSuccess(self, test):
        self.passed += 1
        super().addSuccess(test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='test_*.py')
    args = parser.parse_args(argv)
    report = DiagnosticRun('AUTOMATED TEST RUN')
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests-hermes'), pattern=args.pattern)
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=CountingResult).run(suite)
    failures = result.failures + result.errors
    report.write('AUTOMATED TEST RUN', command=[sys.executable, '-B', 'run_fit_tests.py', '--pattern', args.pattern],
                 total=result.testsRun,
                 passed=result.passed,
                 failed=len(failures), skipped=len(result.skipped),
                 failure_messages=[dict(test=str(test), traceback=trace) for test, trace in failures],
                 skipped_tests=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
                 environment_limitations='Missing dependencies or local socket restrictions appear in failure tracebacks.',
                 test_output=output.getvalue(), benchmarks='Not performed; unit tests use synthetic/mocked data')
    report.finish('tests passed' if result.wasSuccessful() else 'tests failed')
    print('Tests: %s run, %s failures/errors, %s skipped. Diagnostics: %s' %
          (result.testsRun, len(failures), len(result.skipped), report.path))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
