from graphics import dagviz

# example dag
dagviz(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("C", "D"), ("B", "D")], "G1")

# example dg
dagviz(["A", "B", "C", "D"], [("B", "A"), ("A", "C"), ("C", "D"), ("B", "D")], "G2")


dagviz(["A", "B", "C", "D"], [("A", "B"), ("C", "A"), ("C", "D"), ("B", "D")], "G3")
