import tkinter as tk

def calculate():
    try:
        result = eval(display.get())
        display.delete(0, tk.END)
        display.insert(tk.END, str(result))
    except Exception as e:
        display.delete(0, tk.END)
        display.insert(tk.END, "Error")

root = tk.Tk()
root.title("Simple Calculator")

display = tk.Entry(root, width=20, justify='right')
display.grid(row=0, column=0, columnspan=4)

buttons = [
    '7', '8', '9', '/',
    '4', '5', '6', '*',
    '1', '2', '3', '-',
    '0', '.', '=', '+'
]

row_val = 1
col_val = 0

for button in buttons:
    if col_val == 4:
        col_val = 0
        row_val += 1
    tk.Button(root, text=button, width=5, height=2, command=lambda x=button: display.insert(tk.END, x)).grid(row=row_val, column=col_val)
    col_val += 1

calculate_button = tk.Button(root, text='=', width=5, height=2, command=calculate)
calculate_button.grid(row=row_val, column=3)

root.mainloop()