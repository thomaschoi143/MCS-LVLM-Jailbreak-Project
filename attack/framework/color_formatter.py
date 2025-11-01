import logging
import re


class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[94m",  # Blue
        "ERROR": "\033[91m",  # Red
        "DEBUG": "\033[93m",  # Yellow
    }
    RESET = "\033[0m"

    def format(self, record):
        message = super().format(record)
        level = record.levelname

        if level in self.COLORS:
            color = self.COLORS[level]
            if level == "INFO":
                pattern = r"(INFO\s+-\s+(?:[^=]*?[:!])?)"
                match = re.search(pattern, message)
                if match:
                    colored = f"{color}{match.group(1)}{self.RESET}"
                    message = message.replace(match.group(0), colored)
            else:
                message = message.replace(f"{level} - ", f"{color}{level} - {self.RESET}")
        return message
