# The Beginner's Python Handbook

## Chapter 1: Installing Python

Python is a free, beginner-friendly programming language. To install it on Windows, go to python.org, download the latest Python 3 installer, and run it. On the first screen of the installer, tick the box that says "Add python.exe to PATH" before clicking Install Now. If you forget this step, typing `python` in the terminal will not work.

To check your installation, open a terminal and run:

python --version

On macOS and Linux, Python 3 is usually already installed and is called `python3`.

## Chapter 2: Variables and Data Types

A variable is a name that stores a value. You create one with the equals sign, for example `age = 25`. Python figures out the type automatically, so you never write the type yourself.

The most common data types are:
- int: whole numbers such as 10 or -3
- float: decimal numbers such as 3.14
- str: text such as "hello", always written inside quotes
- bool: either True or False
- list: an ordered collection such as [1, 2, 3]
- dict: key-value pairs such as {"name": "Asha", "age": 21}

You can check the type of any value with the built-in function `type()`. For example, `type(3.14)` returns float.

## Chapter 3: Conditions and Loops

Use `if`, `elif` and `else` to make decisions. Python uses indentation (4 spaces) instead of curly braces to show which lines belong to a block. Forgetting to indent correctly causes an IndentationError, which is the most common error beginners see.

A `for` loop repeats code for every item in a collection. The function `range(5)` produces the numbers 0, 1, 2, 3 and 4, so `for i in range(5)` runs five times. A `while` loop keeps running as long as its condition is True; always make sure the condition eventually becomes False, otherwise you create an infinite loop. Press Ctrl+C in the terminal to stop a program stuck in an infinite loop.

Use `break` to leave a loop early and `continue` to skip to the next round of the loop.

## Chapter 4: Functions

A function is a reusable block of code. You define one with the `def` keyword, give it a name, and list its parameters inside brackets. The `return` statement sends a value back to whoever called the function. If a function has no return statement, it returns None.

Parameters can have default values, for example `def greet(name="friend")`. Calling `greet()` without an argument then uses "friend". Keep functions short and give them clear names that describe what they do, such as `calculate_total` instead of `ct`.

## Chapter 5: Virtual Environments and Packages

A virtual environment is a private folder of packages for one project, so different projects do not interfere with each other. Create one with `python -m venv .venv`. On Windows activate it with `.venv\Scripts\activate`; on macOS and Linux use `source .venv/bin/activate`. When it is active, the terminal prompt shows (.venv) at the start.

Install packages with pip, for example `pip install requests`. Save all your project's packages to a file with `pip freeze > requirements.txt`, and install them on another computer with `pip install -r requirements.txt`.

If PowerShell says that running scripts is disabled when you activate the environment, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

## Chapter 6: Handling Errors

Errors that happen while a program is running are called exceptions. Wrap risky code in a `try` block and handle problems in an `except` block, so your program does not crash. For example, converting user input with `int(input())` raises a ValueError if the user types letters.

The `finally` block runs whether an error happened or not, which makes it useful for closing files or database connections. Avoid writing a bare `except:` with no error type, because it hides bugs; catch specific errors like `except ValueError:` instead.

## Chapter 7: Working with Files

Use the `with open("notes.txt", "r", encoding="utf-8") as f:` pattern to read files. The `with` statement automatically closes the file when the block ends, even if an error occurs. Use mode "w" to write (this replaces the file) and mode "a" to append to the end of an existing file. Always pass encoding="utf-8" on Windows to avoid UnicodeDecodeError with special characters.
