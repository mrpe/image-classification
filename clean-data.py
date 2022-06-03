import cv2

clean_data = []

with open("data/full-data.txt", "r") as file:
    content = file.read()
    sample_list = content.split("\n")

for index in range(len(sample_list)):
    try:
        path = sample_list[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320))
        clean_data.append(path)
    except Exception as e:
        print(e)
        print("Error:" + path)
        print("-------------------------------------")

with open("cleaned_data.txt", "w") as file:
    file.write("\n".join(clean_data))
