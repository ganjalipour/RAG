import contextlib
import errno
import importlib.util
import os
import stat
import sys
import tempfile
from ast import literal_eval
from os import path

#####################
#    DIRECTORIES    #
#####################


class DirectoryHelper:
    """A class for handling directory operations such as creating, deleting and getting the contents of a directory."""

    PROJECT_DIR = path.dirname(path.abspath(__file__ + "/../"))
    # Project Directory by using project environment execution path, it is for binary file
    PROJECT_EXECUTION_DIR = path.dirname(sys.executable)[:-10]
    ASSISTANT_DIR = path.join(PROJECT_DIR, "assistant")
    # clients' path
    CLIENT_DIR = path.join(PROJECT_EXECUTION_DIR, "client")
    # Resources directory paths
    RES_DIR = path.join(PROJECT_DIR, "resources")
    MODELS_DIR = path.join(RES_DIR, "models")
    TEXT_DIR = path.join(RES_DIR, "text")
    MEDIA_DIR = path.join(RES_DIR, "media")
    TEST_INPUTS = path.join(MEDIA_DIR, "test_inputs")
    # Storage directory paths
    STORAGE_DIR = PROJECT_DIR + "/storage"

    LOGS_DIR = STORAGE_DIR + "/logs"
    CACHE_DIR = STORAGE_DIR + "/cache"
    CACHE_EMBED_DIR = CACHE_DIR + "/emb_products"
    CACHE_PRODUCT_DIR = CACHE_DIR + "/products"
    CSV_FILE_DIR = STORAGE_DIR + "/csv_files"
    RESPONSES_DIR = f"{STORAGE_DIR}/media/responses"
    FEATURES_DATA_DIR = path.join(STORAGE_DIR, "features_data")

    def __init__(self):
        pass

    @staticmethod
    def get_base_dir() -> str:
        """Gets base directory.

        Returns:
            The OS initial path of the home directory.
        """
        return path.expanduser("~")

    @classmethod
    def is_directory_exists(cls, parent_dir: str, child_dir="") -> bool:
        """Check if the given directory exists

        Args:
           parent_dir: parent directory
           child_dir(str): child directory

        Returns:
             True if path exists, False otherwise
        """

        return path.exists(cls.get_base_dir() + parent_dir + child_dir)

    @classmethod
    def is_directory_exists_as_sub_dir(
        cls, parent_dir_name: str, child_dir_name: str
    ) -> bool:
        """Check if the given directory is a subdirectory of the given parent subdirectories (second child).

        Args:
            parent_dir_name: parent directory name
            child_dir_name: child directory name (subdirectory of the parent directory)

        Returns:
            True if subdirectory, false if not.
        """

        child_dir = f"/{child_dir_name}"
        parent_dir = f"/{parent_dir_name}"

        sub_dirs = os.listdir(parent_dir)

        exist_in_sub_dirs = [
            cls.is_directory_exists(f"{parent_dir}/{sub_dir}", child_dir)
            for sub_dir in sub_dirs
        ]

        return any(exist_in_sub_dirs)

    @classmethod
    def get_app_dir(cls) -> str:
        """Returns project base dir

        Returns:
            Project base dir as string
        """
        return cls.PROJECT_DIR

    @classmethod
    def get_assistant_dir(cls) -> str:
        """Returns assistant dir

        Returns:
            Assistant dir as string
        """
        return cls.get_app_dir() + "/assistant"

    @classmethod
    def get_local_storage_dir(cls) -> str:
        """Returns assistant cache files dir

        Returns:
            Assistant dir as string
        """
        return cls.get_app_dir() + "/.local_storage"

    @classmethod
    def file_exists_in_assistant(cls, file_path: str) -> bool:
        """Checks if giving file exists in assistant files or not

        Args:
            file_path: folder and the file to check its existance

        Returns:
            True if the given file exists, False otherwise
        """
        return os.path.exists(cls.get_assistant_dir() + file_path)

    @staticmethod
    @contextlib.contextmanager
    def in_dir(directory: str) -> None:
        """Change current working directory to `directory`,
        allow the user to run some code, and change back

        Args:
            directory: The path to a directory to work in
        """
        from logger import LOG

        if not os.path.exists(directory):
            LOG.info(
                f"INVAlID PATH | Failed attemp to change current working directory to {directory}"
            )
            return
        current_dir = os.getcwd()
        os.chdir(directory)

        try:
            yield
        finally:
            os.chdir(current_dir)

    @staticmethod
    def guarantee_permissions(
        path_: str, read: bool = True, write: bool = True, execute: bool = True
    ) -> None:
        """Guarantee permissions to the current user to either read, write, or execute

        Args:
            path_: a directory or file path to change its permissions
            read : reading permission state
            write : writing permission state
            execute : execution permission state
        """
        permission = ""
        if read:
            permission += ("|" if permission != "" else "") + str(stat.S_IREAD)
        if write:
            permission += ("|" if permission != "" else "") + str(stat.S_IWRITE)
        if execute:
            permission += ("|" if permission != "" else "") + str(stat.S_IEXEC)
        os.chmod(path_, literal_eval(permission))

    @staticmethod
    def create_dir_if_not_exist(
        directory_path: str, domain: str = None, permission: int = 0o777
    ) -> str:
        """Create the given directory in the desired path.

        Args:
            directory_path: directory path to create.
            domain: Domain. Basically a subdirectory to prevent things like
                          overlapping signal filenames.
            permission: Directory permissions (default is 0o777)

        """
        save = None
        if domain:
            directory_path = os.path.join(directory_path, domain)

        if not path.exists(directory_path):
            try:
                save = os.umask(0)
                os.makedirs(directory_path, permission)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
            finally:
                os.umask(save)

        return directory_path


    @staticmethod
    def get_temp_path(*args) -> str:
        """Generate a valid path in the system temp directory.
        This method accepts one or more strings as arguments. The arguments are
        joined and returned as a complete path inside the systems temp directory.
        Importantly, this will not create any directories or files.
        Example usage: get_temp_path('mycroft', 'audio', 'example.wav')
        Will return the equivalent of: '/tmp/mycroft/audio/example.wav'
        Returns:
            (str) a valid path in the systems temp directory
        """
        from logger import LOG

        try:
            return os.path.join(tempfile.gettempdir(), *args)
        except TypeError as error:
            LOG.error(
                "Could not create a temp path, get_temp_path() only accepts Strings"
            )
            raise TypeError(
                "Could not create a temp path, get_temp_path() only accepts Strings"
            ) from error

    @staticmethod
    def load_attribute_from_module(file_name: str, attribute_name: str) -> dict:
        """Loads a module from a file path and returns a specified attribute from the module.

        Args:
            file_name: The name of the module file.
            attribute_name: The name of the attribute to retrieve.

        Returns:
            The specified attribute from the module.
        """
        file_path = f"{DirectoryHelper.CLIENT_DIR}/{file_name}"
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, attribute_name)
