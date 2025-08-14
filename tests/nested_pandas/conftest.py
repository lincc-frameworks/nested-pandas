import importlib.metadata
import re

import pytest


class TestUtils:
    """Utility helpers for tests around package requirements and version guards.

    Provides helpers to assert that our declared minimum versions for optional/required
    dependencies don't exceed thresholds expected by tests.
    """

    @staticmethod
    def fail_if_min_required_exceeds(dist_name: str, package: str, min_version_to_pass: int) -> None:
        """Fail if the declared minimum required version for `package` exceeds allowed.

        Parameters
        ----------
        dist_name : str
            The installed distribution name to inspect for requirements (e.g., "nested-pandas").
        package : str
            The dependency package name to check (e.g., "pyarrow").
        min_version_to_pass : int
            The highest allowed minimum required a major version. The check fails if the
            declared minimum is strictly greater than this value (e.g., pass 17 to allow
            minimum<=17 and fail on >=18).

        Behavior
        --------
        - Fails via pytest if the package is not found among requirements.
        - Fails via pytest if the declared minimum version is greater than `min_version_to_pass`.
        """
        try:
            import pytest  # imported lazily to avoid hard test dependency at runtime
        except Exception as e:  # pragma: no cover - only used in tests
            raise RuntimeError("pytest is required to use fail_if_min_required_exceeds in tests") from e

        reqs = importlib.metadata.requires(dist_name)
        if reqs is None:
            pytest.fail(f"Could not find `{dist_name}` in `{package}`")
        pkg_lower = package.lower()
        min_required: int | None = None
        for req in reqs:
            # Match the package name at the start of the requirement line (extras/markers may follow)
            if req.lower().startswith(pkg_lower):
                m = re.search(r">=\s*(\d+)", req)
                if m:
                    min_required = int(m.group(1))
                break
        else:
            pytest.fail(
                f"Dependency '{package}' not found in project requirements for distribution '{dist_name}'"
            )

        if min_required is not None and min_required > min_version_to_pass:
            pytest.fail(
                f"Minimum required {package} version is >{min_version_to_pass}; please remove compatibility "
                f"shims and use the newer API directly, then update or delete the tests using this guard."
            )


@pytest.fixture
def test_utils():
    """Pytest fixture exposing the TestUtils helper class for test modules."""
    return TestUtils
