from graphics import dagviz

# example dag
dagviz(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("C", "D"), ("B", "D")], "dag")

# example dg
dagviz(["A", "B", "C", "D"], [("A", "B"), ("C", "A"), ("D", "C"), ("B", "D")], "dg")
