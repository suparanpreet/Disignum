joint_list = [
    # === THUMB ANGLES ===
    [2, 1, 0],  # Thumb CMC joint (landmark 1)
    [3, 2, 1],  # Thumb MCP joint (landmark 2)
    [4, 3, 2],  # Thumb IP joint (landmark 3)
    [1, 0, 5],  # Angle at wrist between thumb and index (landmark 0)

    # === INDEX FINGER ANGLES ===
    [6, 5, 0],  # Index MCP joint (landmark 5)
    [7, 6, 5],  # Index PIP joint (landmark 6)
    [8, 7, 6],  # Index DIP joint (landmark 7)
    [5, 0, 9],  # Angle at wrist between index and middle (landmark 0)

    # === MIDDLE FINGER ANGLES ===
    [10, 9, 0],  # Middle MCP joint (landmark 9)
    [11, 10, 9],  # Middle PIP joint (landmark 10)
    [12, 11, 10],  # Middle DIP joint (landmark 11)
    [9, 0, 13],  # Angle at wrist between middle and ring (landmark 0)

    # === RING FINGER ANGLES ===
    [14, 13, 0],  # Ring MCP joint (landmark 13)
    [15, 14, 13],  # Ring PIP joint (landmark 14)
    [16, 15, 14],  # Ring DIP joint (landmark 15)
    [13, 0, 17],  # Angle at wrist between ring and pinky (landmark 0)

    # === PINKY FINGER ANGLES ===
    [18, 17, 0],  # Pinky MCP joint (landmark 17)
    [19, 18, 17],  # Pinky PIP joint (landmark 18)
    [20, 19, 18],  # Pinky DIP joint (landmark 19)

    # === ADDITIONAL INTER-FINGER ANGLES ===
    [1, 2, 3],  # Alternative thumb angle
    [5, 6, 7],  # Alternative index angle
    [9, 10, 11],  # Alternative middle angle
    [13, 14, 15],  # Alternative ring angle
    [17, 18, 19],  # Alternative pinky angle

    # === FINGER TIP ANGLES ===
    [3, 4, 2],  # Thumb tip angle (landmark 4)
    [7, 8, 6],  # Index tip angle (landmark 8)
    [11, 12, 10],  # Middle tip angle (landmark 12)
    [15, 16, 14],  # Ring tip angle (landmark 16)
    [19, 20, 18],  # Pinky tip angle (landmark 20)

    # === PALM ANGLES ===
    [1, 0, 17],  # Thumb to pinky spread at wrist
    [5, 0, 13],  # Index to ring spread at wrist
    [9, 0, 1],  # Middle to thumb at wrist
    [17, 0, 1],  # Pinky to thumb at wrist
]