for i in range(4):
    with open('template_test.py', "rt") as fin:
        with open(f"Script{i}.py", "wt") as fout:
            for line in fin:
                fout.write(line.replace("inp", str(i)))