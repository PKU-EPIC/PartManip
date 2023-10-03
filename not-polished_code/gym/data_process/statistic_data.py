from pathlib import Path
##door###
print("door")
asset_order_all = []
train_asset_root = Path("assets/door/train")
valInter_asset_root = Path("assets/door/valInter")
valIntra_asset_root = Path("assets/door/valIntra")
total_train_asset_paths = list(train_asset_root.iterdir())
total_valIntra_asset_paths = list(valIntra_asset_root.iterdir())
total_valInter_asset_paths = list(valInter_asset_root.iterdir())
new_path = total_train_asset_paths 
path_to_remove = [
    Path("assets/door/train/StorageFurniture-46037-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-41083-link_1-handle_5-joint_1-handlejoint_5"),
]

for p in path_to_remove:
    if p in new_path:
        new_path.remove(p)
total_train_asset_paths = new_path

train_category = {}
train = 0
for p in total_train_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in train_category:
        train_category[p_category] = 0
    train_category[p_category] += 1
    train+= 1

print(train_category, train)

valIntra_category = {}
val = 0
for p in total_valIntra_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valIntra_category:
        valIntra_category[p_category] = 0
    valIntra_category[p_category] += 1
    val += 1

print(valIntra_category, val)

valInter_category = {}
val = 0
for p in total_valInter_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valInter_category:
        valInter_category[p_category] = 0
    valInter_category[p_category] += 1
    val += 1

print(valInter_category, val)

for p in total_train_asset_paths:
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valIntra_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valInter_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

print("drawer")
###drawer###
train_asset_root = Path("assets/drawer/train")
valIntra_asset_root = Path("assets/drawer/valIntra")
valInter_asset_root = Path("assets/drawer/valInter")
total_train_asset_paths = list(train_asset_root.iterdir())
total_valIntra_asset_paths = list(valIntra_asset_root.iterdir())
total_valInter_asset_paths = list(valInter_asset_root.iterdir())

new_path = total_train_asset_paths 
path_to_remove = [
    Path("assets/drawer/train/StorageFurniture-48855-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/Dishwasher-12085-link_1-handle_0-joint_1-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-47235-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-48169-link_0-handle_0-joint_0-handlejoint_0")]

asset_to_remove = ["47207", "46537", "19855", "30666"]
for p in path_to_remove:
    if p in new_path:
        new_path.remove(p)
        continue
for a in asset_to_remove:
    for p in new_path:
        if a in str(p):
            new_path.remove(p)
for a in asset_to_remove:
    for p in total_valIntra_asset_paths:
        if a in str(p):
            total_valIntra_asset_paths.remove(p)
for a in asset_to_remove:
    for p in total_valInter_asset_paths:
        if a in str(p):
            total_valInter_asset_paths.remove(p)

total_train_asset_paths = new_path

train_category = {}
train = 0
for p in total_train_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in train_category:
        train_category[p_category] = 0
    train_category[p_category] += 1
    train+= 1

print(train_category, train)

valIntra_category = {}
val = 0
for p in total_valIntra_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valIntra_category:
        valIntra_category[p_category] = 0
    valIntra_category[p_category] += 1
    val += 1

print(valIntra_category, val)

valInter_category = {}
val = 0
for p in total_valInter_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valInter_category:
        valInter_category[p_category] = 0
    valInter_category[p_category] += 1
    val += 1

print(valInter_category, val)

for p in total_train_asset_paths:
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valIntra_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valInter_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

print("button")
###button###
train_asset_root = Path("assets/button/train")
valIntra_asset_root = Path("assets/button/valIntra")
valInter_asset_root = Path("assets/button/valInter")
total_train_asset_paths = list(train_asset_root.iterdir())
total_valIntra_asset_paths = list(valIntra_asset_root.iterdir())
total_valInter_asset_paths = list(valInter_asset_root.iterdir())

new_path = total_train_asset_paths 
path_to_remove = []
asset_to_remove = ["103452", "103351", "103425"]
for p in path_to_remove:
    if p in new_path:
        new_path.remove(p)
        continue
for a in asset_to_remove:
    for p in new_path:
        if a in str(p):
            new_path.remove(p)
for a in asset_to_remove:
    for p in total_valIntra_asset_paths:
        if a in str(p):
            total_valIntra_asset_paths.remove(p)
for a in asset_to_remove:
    for p in total_valInter_asset_paths:
        if a in str(p):
            total_valInter_asset_paths.remove(p)

total_train_asset_paths = new_path

train_category = {}
train = 0
for p in total_train_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in train_category:
        train_category[p_category] = 0
    train_category[p_category] += 1
    train+= 1

print(train_category, train)

valIntra_category = {}
val = 0
for p in total_valIntra_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valIntra_category:
        valIntra_category[p_category] = 0
    valIntra_category[p_category] += 1
    val += 1

print(valIntra_category, val)

valInter_category = {}
val = 0
for p in total_valInter_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valInter_category:
        valInter_category[p_category] = 0
    valInter_category[p_category] += 1
    val += 1

print(valInter_category, val)
for p in total_train_asset_paths:
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valIntra_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valInter_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

print("handle")
###handle###
train_asset_root = Path("assets/handle/train")
valIntra_asset_root = Path("assets/handle/valIntra")
valInter_asset_root = Path("assets/handle/valInter")
total_train_asset_paths = list(train_asset_root.iterdir())
total_valIntra_asset_paths = list(valIntra_asset_root.iterdir())
total_valInter_asset_paths = list(valInter_asset_root.iterdir())

new_path = total_train_asset_paths 
path_to_remove = []
#"103543", "7221", "25493", "7310", 
asset_to_remove = ["47207", "46537", "19855", "30666"]
for p in path_to_remove:
    if p in new_path:
        new_path.remove(p)
        continue
for a in asset_to_remove:
    for p in new_path:
        if a in str(p):
            new_path.remove(p)
for a in asset_to_remove:
    for p in total_valIntra_asset_paths:
        if a in str(p):
            total_valIntra_asset_paths.remove(p)
for a in asset_to_remove:
    for p in total_valInter_asset_paths:
        if a in str(p):
            total_valInter_asset_paths.remove(p)

total_train_asset_paths = new_path

train_category = {}
train = 0
for p in total_train_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in train_category:
        train_category[p_category] = 0
    train_category[p_category] += 1
    train+= 1

print(train_category, train)

valIntra_category = {}
val = 0
for p in total_valIntra_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valIntra_category:
        valIntra_category[p_category] = 0
    valIntra_category[p_category] += 1
    val += 1

print(valIntra_category, val)

valInter_category = {}
val = 0
for p in total_valInter_asset_paths:
    p_category = str(p).split("/")[-1].split("-")[0]
    if p_category not in valInter_category:
        valInter_category[p_category] = 0
    valInter_category[p_category] += 1
    val += 1

print(valInter_category, val)

for p in total_train_asset_paths:
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valIntra_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

for p in total_valInter_asset_paths :
    if str(p).split('-')[1] not in asset_order_all:
        asset_order_all.append(str(p).split('-')[1])

print(asset_order_all)
print(len(asset_order_all))