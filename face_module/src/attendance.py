def mark_attendance(recognized_names):

    present_students = []

    for name in recognized_names:

        if name != "Unknown" and name not in present_students:
            present_students.append(name)

    print("\n========== ATTENDANCE ==========")

    for name in present_students:
        print(f"{name}: Present")

    print("================================")

    return present_students