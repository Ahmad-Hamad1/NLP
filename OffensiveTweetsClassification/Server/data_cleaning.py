with open("L-HSAB.txt", encoding='utf-8') as f:
    lines = f.readlines()

with open("data.txt", "w", encoding='utf-8') as f:
    for idx, line in enumerate(lines):
        if len(line) == 0:
            continue
        tempLine = ' '.join(line.split())
        tempLine = tempLine.replace(",", " ")
        tempLine = tempLine.strip()
        lastOccur = tempLine.rfind(" ")
        tempLine = tempLine[:lastOccur] + "," + tempLine[lastOccur + 1:]
        tempLine = tempLine.replace("normal", "0")
        tempLine = tempLine.replace("abusive", "1")
        tempLine = tempLine.replace("hate", "1")
        f.write(tempLine)
        f.write("\n")
