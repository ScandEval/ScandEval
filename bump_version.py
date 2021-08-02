'''Script that fetches and bumps versions'''

from pathlib import Path
import re
from typing import Union, Tuple
import subprocess
import datetime as dt


PACKAGE_NAME = 'scandeval'


def get_current_version(return_tuple: bool = False
                        ) -> Union[str, Tuple[int, int, int]]:
    '''Fetch the current version without importing __init__.py.

    Args:
        return_tuple (bool, optional):
            Whether to return a tuple of three numbers, corresponding to the
            major, minor and patch version. Defaults to False.

    Returns:
        str or tuple of three ints: The current version
    '''
    init_file = Path(PACKAGE_NAME) / '__init__.py'
    init = init_file.read_text()
    version_regex = r"(?<=__version__ = ')[0-9]+\.[0-9]+\.[0-9]+(?=')"
    version = re.search(version_regex, init)[0]
    if return_tuple:
        major, minor, patch = [int(v) for v in version.split('.')]
        return (major, minor, patch)
    else:
        return version


def set_new_version(major: int, minor: int, patch: int):
    '''Sets a new version.

    Args:
        major (int):
            The major version. This only changes when the code stops being
            backwards compatible.
        minor (int):
            The minor version. This changes when a backwards compatible change
            happened.
        patch (init):
            The patch version. This changes when the only new changes are bug
            fixes.
    '''
    version = f'{major}.{minor}.{patch}'

    # Get current changelog and ensure that it has an [Unreleased] entry
    changelog_path = Path('CHANGELOG.md')
    changelog = changelog_path.read_text()
    if '[Unreleased]' not in changelog:
        raise RuntimeError('No [Unreleased] entry in CHANGELOG.md.')

    # Add version to CHANGELOG
    today = dt.date.today().strftime('%Y-%m-%d')
    new_changelog = re.sub(r'\[Unreleased\].*', f'[v{version}] - {today}',
                           changelog)
    changelog_path.write_text(new_changelog)

    # Get current __init__.py content
    init_file = Path(PACKAGE_NAME) / '__init__.py'
    init = init_file.read_text()

    # Replace __version__ in __init__.py with the new one
    version_regex = r"(?<=__version__ = ')[0-9]+\.[0-9]+\.[0-9]+(?=')"
    new_init = re.sub(version_regex, version, init)
    with init_file.open('w') as f:
        f.write(new_init)

    # Get current Sphinx conf.py content
    sphinx_conf_file = Path('docs') / 'source' / 'conf.py'
    sphinx_conf = sphinx_conf_file.read_text()

    # Replace `release` in conf.py with the new one
    version_regex = r"(?<=release = ')v[0-9]+\.[0-9]+\.[0-9]+(?=')"
    new_sphinx_conf = re.sub(version_regex, version, sphinx_conf)
    with sphinx_conf_file.open('w') as f:
        f.write(new_sphinx_conf)

    # Add to version control
    subprocess.run(['git', 'add', str(Path(PACKAGE_NAME) / '__init__.py')])
    subprocess.run(['git', 'add', 'CHANGELOG.md'])
    subprocess.run(['git', 'add', 'docs/source/conf.py'])
    subprocess.run(['git', 'commit', '-m', f'feat: v{version}'])
    subprocess.run(['git', 'tag', f'v{version}'])


def bump_major():
    '''Add one to the major version'''
    major, minor, patch = get_current_version(return_tuple=True)
    set_new_version(major + 1, 0, 0)


def bump_minor():
    '''Add one to the minor version'''
    major, minor, patch = get_current_version(return_tuple=True)
    set_new_version(major, minor + 1, 0)


def bump_patch():
    '''Add one to the patch version'''
    major, minor, patch = get_current_version(return_tuple=True)
    set_new_version(major, minor, patch + 1)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--major', const=True, nargs='?', default=False,
                        help='Bump the major version by one.')
    parser.add_argument('--minor', const=True, nargs='?', default=False,
                        help='Bump the minor version by one.')
    parser.add_argument('--patch', const=True, nargs='?', default=False,
                        help='Bump the patch version by one.')
    args = parser.parse_args()

    if args.major + args.minor + args.patch != 1:
        raise RuntimeError('Exactly one of --major, --minor and --patch must '
                           'be selected.')
    elif args.major:
        bump_major()
    elif args.minor:
        bump_minor()
    elif args.patch:
        bump_patch()
