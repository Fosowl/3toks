"""One persistent Chrome profile, shared by browser sessions one at a time.

Bot checks trust a browser that comes back with its cookies: a solved
Cloudflare check, an accepted consent banner and the site's own cookies
all live in the Chrome profile. A fresh throwaway profile per session loses
them and makes every visit look like a first, suspicious one.

Chrome refuses to start on a profile that another Chrome uses. So the
fetcher first takes an exclusive OS file lock in the shared profile. The
kernel drops the lock when the process ends, so a crash leaves no stale
lock. When another 3toks process holds the lock, this session uses a
temporary profile and deletes it on release.
"""
import os
import shutil
import tempfile

PROFILE_DIR_NAME = "chrome-profile"
APP_DIR_NAME = "threetoks"
LOCK_FILE_NAME = "threetoks.lock"
TEMP_PREFIX = "threetoks_chrome_"
CACHE_ENV_VAR = "XDG_CACHE_HOME"
WINDOWS_CACHE_ENV_VAR = "LOCALAPPDATA"


def default_profile_dir() -> str:
    """Return the path of the shared Chrome profile.

    The browser fetcher uses this path so cookies survive between sessions.
    ``ChromeProfile`` calls it when no directory is given.
    It follows the per-user cache folder of the operating system.

    Returns:
        str: ``$XDG_CACHE_HOME/threetoks/chrome-profile`` (default
        ``~/.cache``), or ``%LOCALAPPDATA%\\threetoks\\chrome-profile``.
    """
    env_var = WINDOWS_CACHE_ENV_VAR if os.name == "nt" else CACHE_ENV_VAR
    base = os.environ.get(env_var) or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, APP_DIR_NAME, PROFILE_DIR_NAME)


def _lock_exclusive(descriptor: int) -> None:
    """Take a non-blocking exclusive lock on an open file.

    ``_try_lock`` uses this to make sure only one Chrome uses the profile.
    It calls it with the descriptor of the lock file.
    It raises ``OSError`` at once when another process holds the lock.

    Args:
        descriptor (int): OS file descriptor of the open lock file.
    """
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        return
    import fcntl
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _try_lock(directory: str):
    """Lock the profile directory for this process, if it is free.

    ``ChromeProfile.claim`` uses this before it gives the shared profile out.
    It calls it with the shared profile path.
    It creates the directory and the lock file, then tries the lock.

    Args:
        directory (str): the shared Chrome profile directory.

    Returns:
        file | None: the open lock file (keep it open to hold the lock),
        or None when another process holds the lock or the directory
        cannot be written.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        handle = open(os.path.join(directory, LOCK_FILE_NAME), "a+")
    except OSError:
        return None
    try:
        _lock_exclusive(handle.fileno())
    except OSError:
        handle.close()
        return None
    return handle


class ChromeProfile:
    """The Chrome user-data-dir of one browser session.

    ``claim`` gives the shared profile when it is free, else a temporary
    one. ``release`` drops the lock and deletes a temporary profile. Both
    are safe to call again.
    """

    def __init__(self, shared_dir: str | None = None):
        """Store the shared profile path; nothing is locked yet.

        ``BrowserFetcher`` makes one profile object per fetcher.
        Tests give a temporary directory, production code gives nothing.
        It only stores the path.

        Args:
            shared_dir (str | None): the shared profile directory, or None
                for ``default_profile_dir()``.
        """
        self.shared_dir = shared_dir
        self.path: str | None = None
        self._lock = None
        self._is_temporary = False

    def claim(self) -> str:
        """Return the profile directory for Chrome, locking it if shared.

        ``BrowserFetcher`` calls this before it starts Chrome.
        A second call returns the same path without a new lock.
        It uses the shared profile when free, else a new temporary one.

        Returns:
            str: the directory to pass as ``--user-data-dir``.
        """
        if self.path is not None:
            return self.path
        shared = self.shared_dir or default_profile_dir()
        self._lock = _try_lock(shared)
        if self._lock is not None:
            self.path = shared
        else:
            self.path = tempfile.mkdtemp(prefix=TEMP_PREFIX)
            self._is_temporary = True
        return self.path

    def release(self) -> None:
        """Drop the lock, and delete the profile if it was temporary.

        ``BrowserFetcher.close`` calls this after Chrome quits.
        It takes no data and is safe to call again.
        It makes the shared profile free for the next session.
        """
        if self._lock is not None:
            self._lock.close()
            self._lock = None
        if self._is_temporary and self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
        self.path = None
        self._is_temporary = False


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        shared = os.path.join(root, "profile")
        first, second = ChromeProfile(shared), ChromeProfile(shared)
        assert first.claim() == shared
        fallback = second.claim()
        assert fallback != shared and os.path.isdir(fallback)
        second.release()
        assert not os.path.exists(fallback), "temporary profile must go"
        first.release()
        assert ChromeProfile(shared).claim() == shared, "lock is released"
    print("smoke OK")
