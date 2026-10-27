"""Audit two hand-specified scene graphs; this does not run Boogu-Image.

The graphs encode intended scene contents, not information extracted from
generated pixels. Run with the Python standard library only.
"""


def audit(scene):
    circles = [x for x in scene["objects"] if x["shape"] == "circle"]
    squares = [x for x in scene["objects"] if x["shape"] == "square"]
    return {
        "three red circles": len(circles) == 3 and all(x["color"] == "red" for x in circles),
        "one blue square": len(squares) == 1 and squares[0]["color"] == "blue",
        "circles left of square": len(squares) == 1 and bool(circles)
        and all(x["x"] < squares[0]["x"] for x in circles),
        "title is seven times eight": scene["title"] == str(7 * 8),
        "white background": scene["background"] == "white",
    }


def make_scene(count=3, background="white", title="56"):
    return {
        "objects": [
            {"shape": "circle", "color": "red", "x": x}
            for x in [0.15, 0.30, 0.45, 0.60][:count]
        ] + [{"shape": "square", "color": "blue", "x": 0.80}],
        "title": title,
        "background": background,
    }


if __name__ == "__main__":
    faithful = audit(make_scene())
    drifting = audit(make_scene(count=4, background="navy", title="ORBIT"))
    assert all(faithful.values())
    assert sum(drifting.values()) == 2
    print("Faithful specification:", faithful)
    print("Drifting specification:", drifting)
    print("These checks validate symbolic specifications, not image generation.")
