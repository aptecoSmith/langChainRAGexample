from colorama import init, Fore, Style
import textwrap


class ColorfulPrinterColorama:
    def __init__(self):
        # Initialize colorama to auto-reset the style
        init(autoreset=True)

    def question_print(self, input, width=200):
        """Prints the input text in red."""
        wrapped_text = textwrap.fill(input, width=width)
        print(Fore.RED+ Style.BRIGHT + wrapped_text)

    def answer_print(self, input, width=200):
        """Prints the input text in green."""
        wrapped_text = textwrap.fill(input, width=width)
        print(Fore.GREEN + wrapped_text)



