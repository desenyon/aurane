from aurane.parser import Parser

with open("tests/broken.aur", "r") as f:
    source = f.read()

parser = Parser(source)
program = parser.parse()

print(f"Models found: {[m.name for m in program.models]}")
print(f"Errors found: {len(parser.errors)}")
for line, msg in parser.errors:
    print(f"  Line {line}: {msg}")
