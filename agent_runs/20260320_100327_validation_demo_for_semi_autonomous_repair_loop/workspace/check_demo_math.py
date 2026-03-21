from demo_math import add


def main() -> None:
    assert add(2, 3) == 5
    assert add(-2, 7) == 5
    print("validation checks passed")


if __name__ == "__main__":
    main()
