"""
This module defines a custom exception class that captures detailed error information
and a utility function to format error messages with file name, line number, and error message.
"""

import logging
import sys


def error_message_detail(error, error_detail: sys):
    """
    Generates a detailed error message.

    Args:
        error (Exception): The exception object.
        error_detail (module): The sys module, used to get exception details.

    Returns:
        str: Formatted error message with file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_message


class CustomException(Exception):
    """
    Custom exception class that captures detailed error information.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException instance.

        Args:
            error_message (str): The error message.
            error_detail (module): The sys module, used to get exception details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    pass
