import tarfile
from .utils import on_progress, get_file_progress_file_object_class, ProgressFileObject


def extract_tgz_file(tgz_filename:str) -> None:
    """
    Helper function to speed up extraction of tar files.
    Extracts files into the same directory as the tar file

    Args:
    tgz_filename - tar file name in directory

    Returns:
    None
    """
    tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)
    tar = tarfile.open(fileobj=ProgressFileObject(tgz_filename))
    tar.extractall()
    tar.close()