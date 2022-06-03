import cv2

def camera_indices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def choose_index():
    choice = -1
    indices = camera_indices()
    # Show the user how each camera looks:
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        # Allow user to select this camera or continue
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)

            cv2.imshow(f'Camera Index: {idx}', image)
            key = cv2.waitKey(0)
            print(key)
            if key == 121:# ('y')
                choice = idx
                cap.release()
                cv2.destroyAllWindows()
            if key != 121:
                cap.release()
                cv2.destroyAllWindows()
        
        if choice == idx: # i.e. a choice has been made
            break

    return choice




if __name__ == '__main__':
    index_choice = choose_index()
    print(f'You chose camera index: {index_choice}')
    # Write the index to a file
    with open('camera_index.txt', 'w') as f:
        f.write(str(index_choice))

    cv2.destroyAllWindows()
