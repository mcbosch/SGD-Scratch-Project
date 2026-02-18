import time
import sys


toolbar_width = 100

def one_progress_bar(toolbar_width):
    # setup toolbar
    sys.stdout.write("Charging:\t")
    sys.stdout.flush()

    for i in range(toolbar_width):
        time.sleep(0.1) # do real work here
        perc = int((i+1)/toolbar_width*100)

        # update the bar
        sys.stdout.write(f"{perc:02d}%|\033[32;7m" + " " * perc + "\033[0m" + " " * (100-perc)+"|" )
        sys.stdout.flush()
        sys.stdout.write("\b" * (100+2+3))

    sys.stdout.write("\n") # this ends the progress bar

import sys
import time

def two_progress_bar(n1, n2):
    bar_width = 40  # más manejable que 100

    # Dibuja las dos líneas iniciales
    sys.stdout.write(f"Progress 1:\t{'':40s}\n")
    sys.stdout.write(f"\tProgress 2:\t{'':40s}\n")
    sys.stdout.flush()

    for i in range(n1):
        # Subir 2 líneas para redibujar barra 1
        sys.stdout.write("\033[2A")
        p1 = int(bar_width * i / n1)
        sys.stdout.write(f"\rProgress 1:\t\033[31;7m{' '*p1}\033[0m{' '*(bar_width-p1)}| {i:02d}/{n1}\n")
        sys.stdout.flush()

        for j in range(n2):
            time.sleep(0.05)
            p2 = int(bar_width * j / n2)
            sys.stdout.write(f"\r\tProgress 2:\t\033[32;7m{' '*p2}\033[0m{' '*(bar_width-p2)}| {j:02d}/{n2}")
            sys.stdout.flush()

        # Barra 2 completa al terminar el inner loop
        sys.stdout.write(f"\r\tProgress 2:\t\033[32;7m{' '*bar_width}\033[0m| {n2:02d}/{n2}\n")
        sys.stdout.flush()
        time.sleep(0.05)

    # Barra 1 completa
    sys.stdout.write("\033[2A")
    sys.stdout.write(f"\rProgress 1:\t\033[31;7m{' '*bar_width}\033[0m| {n1:02d}/{n1}\n\n")
    sys.stdout.flush()
two_progress_bar(100,100)