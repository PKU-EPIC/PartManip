from isaacgym import gymapi
import numpy as np
from pathlib import Path
# [
#     Path('assets/handle/valIntra/TrashCan-103435-link_0-handle_0-joint0-handlejoint_0'), 
#     Path('assets/handle/valIntra/TrashCan-103542-link_1-handle_3-joint1-handlejoint_3'), 
#     Path('assets/handle/valIntra/TrashCan-103453-link_0-handle_0-joint0-handlejoint_0'), 
#     Path('assets/handle/valIntra/TrashCan-102252-link_1-handle_3-joint1-handlejoint_3'),
# ]

[
    Path('assets/handle/train/TrashCan-104525-link_1-handle_3-joint_1-handlejoint_3'), 
    Path('assets/handle/train/TrashCan-102194-link_2-handle_4-joint_2-handlejoint_4'), 
    Path('assets/handle/train/TrashCan-102996-link_0-handle_0-joint_0-handlejoint_0'), 
    Path('assets/handle/train/TrashCan-102244-link_1-handle_3-joint_1-handlejoint_3'), 
    Path('assets/handle/train/TrashCan-102234-link_1-handle_3-joint_1-handlejoint_3'), 
    Path('assets/handle/train/TrashCan-103543-link_1-handle_3-joint_1-handlejoint_3'), 
    Path('assets/handle/train/TrashCan-103595-link_2-handle_4-joint_2-handlejoint_4'), 
    Path('assets/handle/train/TrashCan-103013-link_0-handle_0-joint_0-handlejoint_0'),
    Path('assets/handle/valIntra/TrashCan-103435-link_0-handle_0-joint_0-handlejoint_0'), 
    Path('assets/handle/valIntra/TrashCan-103542-link_1-handle_3-joint_1-handlejoint_3'), 
    Path('assets/handle/valIntra/TrashCan-103453-link_0-handle_0-joint_0-handlejoint_0'), 
    Path('assets/handle/valIntra/TrashCan-102252-link_1-handle_3-joint_1-handlejoint_3'),
]
# asset to use
assets_to_use = [
    # Path('assets/handle/train/TrashCan-104525-link_1-handle_3-joint_1-handlejoint_3'), 
    # Path('assets/handle/train/TrashCan-102194-link_2-handle_4-joint_2-handlejoint_4'), 
    # Path('assets/handle/train/TrashCan-102996-link_0-handle_0-joint_0-handlejoint_0'), 
    # Path('assets/handle/train/TrashCan-102244-link_1-handle_3-joint_1-handlejoint_3'), 
    # Path('assets/handle/train/TrashCan-102234-link_1-handle_3-joint_1-handlejoint_3'), 
    # Path('assets/handle/train/TrashCan-103543-link_1-handle_3-joint_1-handlejoint_3'), 
    # Path('assets/handle/train/TrashCan-103595-link_2-handle_4-joint_2-handlejoint_4'), 
    # Path('assets/handle/train/TrashCan-103013-link_0-handle_0-joint_0-handlejoint_0'),
    # Path('assets/handle/valIntra/TrashCan-103435-link_0-handle_0-joint_0-handlejoint_0'), 
    # Path('assets/handle/valIntra/TrashCan-103542-link_1-handle_3-joint_1-handlejoint_3'), 
    # Path('assets/handle/valIntra/TrashCan-103453-link_0-handle_0-joint_0-handlejoint_0'), 
    # Path('assets/handle/valIntra/TrashCan-102252-link_1-handle_3-joint_1-handlejoint_3'),
] # empty -> all

assets_to_imitate=[
    Path("assets/door/train/Dishwasher-12543-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46120-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/Dishwasher-12587-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46134-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/Dishwasher-12606-link_1-handle_0-joint_1-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46134-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-41003-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46199-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-41085-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46700-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-41452-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46825-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45162-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46859-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45176-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46874-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45189-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46955-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45189-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-47227-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45244-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47290-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45271-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-47290-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-45271-link_5-handle_15-joint_5-handlejoint_15"),
    Path("assets/door/train/StorageFurniture-47290-link_5-handle_15-joint_5-handlejoint_15"),
    Path("assets/door/train/StorageFurniture-45332-link_1-handle_5-joint_1-handlejoint_5"),
    Path("assets/door/train/StorageFurniture-47595-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45354-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47595-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-45354-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47669-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-45378-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47669-link_5-handle_15-joint_5-handlejoint_15"),
    Path("assets/door/train/StorageFurniture-45384-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47701-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45387-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-47853-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-45397-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-47926-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/door/train/StorageFurniture-45420-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47944-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45448-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47976-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45463-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48018-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45463-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-48018-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45503-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/door/train/StorageFurniture-48177-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-45575-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48356-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45696-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48356-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-45696-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-48379-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45696-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-48379-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45749-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-48700-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45936-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48700-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45948-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-49025-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-46033-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-49062-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-46057-link_1-handle_0-joint_1-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-49062-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-46108-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-49062-link_5-handle_15-joint_5-handlejoint_15"),
]
assets_to_imitate_ = [
    Path("assets/door/train/StorageFurniture-49062-link_5-handle_15-joint_5-handlejoint_15"),
    Path("assets/door/train/StorageFurniture-46490-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46490-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46490-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47669-link_5-handle_15-joint_5-handlejoint_15"),
    Path("assets/door/train/StorageFurniture-45632-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-45687-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46480-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48379-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48379-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45238-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45749-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45378-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46037-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45420-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48878-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45420-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-48356-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-48356-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47185-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45594-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45594-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45387-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45612-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-46092-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/Dishwasher-12606-link_1-handle_0-joint_1-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-49133-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-47944-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47388-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47281-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45189-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45162-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45189-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-45936-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47817-link_1-handle_0-joint_1-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45948-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-49025-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-49025-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-45622-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45948-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45622-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46874-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45384-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46825-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48855-link_2-handle_3-joint_2-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46874-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46480-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-45696-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45696-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47853-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-45696-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-47853-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47853-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/Dishwasher-12480-link_1-handle_0-joint_1-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-40417-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-46944-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47747-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46108-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-49042-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-46108-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-46825-link_1-handle_5-joint_1-handlejoint_5"),
    Path("assets/door/train/StorageFurniture-47669-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-45523-link_2-handle_5-joint_2-handlejoint_5"),
    Path("assets/door/train/StorageFurniture-46120-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-45780-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-47669-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46700-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-41003-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-46002-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-41003-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47669-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-47595-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-47976-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-49062-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-47701-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45378-link_1-handle_5-joint_1-handlejoint_5"),
    Path("assets/door/train/StorageFurniture-48855-link_1-handle_2-joint_1-handlejoint_2"),
    Path("assets/door/train/StorageFurniture-47701-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-46616-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48859-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48859-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45448-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48063-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45612-link_3-handle_3-joint_3-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45780-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46120-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45662-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45271-link_5-handle_15-joint_5-handlejoint_15"),
    Path("assets/door/train/StorageFurniture-46199-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46134-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-45661-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45853-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47701-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-46408-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47290-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45638-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45235-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47024-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45332-link_1-handle_5-joint_1-handlejoint_5"),
    Path("assets/door/train/StorageFurniture-47613-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47290-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-47613-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45575-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48700-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45463-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-48700-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45463-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46456-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45194-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-49062-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45696-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-41083-link_1-handle_5-joint_1-handlejoint_5"),
    Path("assets/door/train/StorageFurniture-46134-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45503-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-46002-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47976-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47595-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47976-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-47595-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-46002-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-46417-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-41085-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-47669-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-41085-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-46856-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46856-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47278-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-47926-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/door/train/Dishwasher-12579-link_1-handle_0-joint_1-handlejoint_0      "),
    Path("assets/door/train/StorageFurniture-45387-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-48513-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/Dishwasher-12543-link_0-handle_0-joint_0-handlejoint_0      "),
    Path("assets/door/train/StorageFurniture-45964-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/Microwave-7310-link_0-handle_0-joint_0-handlejoint_0        "),
    Path("assets/door/train/StorageFurniture-47926-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-46236-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47290-link_5-handle_15-joint_5-handlejoint_15"),
    Path("assets/door/train/StorageFurniture-46401-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46732-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47088-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-48177-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-46732-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46180-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47601-link_1-handle_0-joint_1-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45189-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/Dishwasher-12553-link_0-handle_0-joint_0-handlejoint_0        "),
    Path("assets/door/train/StorageFurniture-45271-link_4-handle_12-joint_4-handlejoint_12"),
    Path("assets/door/train/StorageFurniture-47742-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47529-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45948-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-46801-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45503-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/door/train/StorageFurniture-41083-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-49025-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-48177-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45354-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45305-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45354-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45305-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45176-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45423-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45423-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46955-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45505-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-45670-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-45001-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48018-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-47227-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-48018-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47227-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45332-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48018-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-46847-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-40417-link_5-handle_5-joint_5-handlejoint_5"),
    Path("assets/door/train/StorageFurniture-45271-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-45749-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-47577-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45749-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/door/train/StorageFurniture-40147-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47577-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-48356-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/door/train/StorageFurniture-46197-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47808-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/door/train/StorageFurniture-47808-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/door/train/StorageFurniture-45397-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/door/train/StorageFurniture-45146-link_1-handle_1-joint_1-handlejoint_1"),
]
assets_to_use_easy = [
    #dishwasher 17
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12092-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12259-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12480-link_1-handle_0-joint_1-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12531-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12540-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12543-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12553-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12561-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12579-link_1-handle_0-joint_1-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12580-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12583-link_1-handle_0-joint_1-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12587-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12590-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12592-link_1-handle_0-joint_1-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12605-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12606-link_1-handle_0-joint_1-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Dishwasher-12614-link_1-handle_0-joint_1-handlejoint_0"),
    #microwave 4
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Microwave-7167-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Microwave-7221-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Microwave-7310-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/Microwave-7320-link_0-handle_0-joint_0-handlejoint_0"),
    #storage_furniture
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-40147-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-41003-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-41003-link_1-handle_3-joint_1-handlejoint_3"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-41003-link_2-handle_6-joint_2-handlejoint_6"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-41003-link_3-handle_9-joint_3-handlejoint_9"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-41083-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-41083-link_1-handle_5-joint_1-handlejoint_5"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-41452-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-44781-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-44781-link_1-handle_3-joint_1-handlejoint_3"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45001-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45001-link_1-handle_3-joint_1-handlejoint_3"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45087-link_1-handle_0-joint_1-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45146-link_1-handle_1-joint_1-handlejoint_1"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45162-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45176-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45194-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45194-link_1-handle_3-joint_1-handlejoint_3"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45203-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45213-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45238-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45238-link_1-handle_3-joint_1-handlejoint_3"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45244-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45249-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45267-link_0-handle_0-joint_0-handlejoint_0"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45271-link_2-handle_6-joint_2-handlejoint_6"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45271-link_3-handle_9-joint_3-handlejoint_9"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45271-link_4-handle_12-joint_4-handlejoint_12"),
    Path("/data2/haoran/RL-Pose/PoseOrientedGym/assets/door/train/StorageFurniture-45271-link_5-handle_15-joint_5-handlejoint_15"),
]

assets_to_use_drawer = [
    Path("assets/drawer/train/StorageFurniture-47565-link_2-handle_3-joint_2-handlejoint_3"),
    Path("assets/drawer/train/StorageFurniture-45194-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/drawer/train/StorageFurniture-46060-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/drawer/train/StorageFurniture-46462-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/drawer/train/StorageFurniture-41083-link_2-handle_10-joint_2-handlejoint_10"),
    
    Path("assets/drawer/train/StorageFurniture-45219-link_3-handle_3-joint_3-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-45243-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-45271-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-45271-link_1-handle_3-joint_1-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-45710-link_3-handle_15-joint_3-handlejoint_15"),
    
    #Path("assets/drawer/train/StorageFurniture-46334-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-47207-link_2-handle_6-joint_2-handlejoint_6"),
    #Path("assets/drawer/train/StorageFurniture-46466-link_2-handle_2-joint_2-handlejoint_2"),
    #Path("assets/drawer/train/StorageFurniture-48169-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-47296-link_1-handle_1-joint_1-handlejoint_1"),

    #Path("assets/drawer/train/StorageFurniture-45290-link_2-handle_6-joint_2-handlejoint_6"),
    #Path("assets/drawer/train/StorageFurniture-46060-link_2-handle_6-joint_2-handlejoint_6"),
    #Path("assets/drawer/train/StorageFurniture-48169-link_2-handle_4-joint_2-handlejoint_4"),
    #Path("assets/drawer/train/StorageFurniture-45801-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/drawer/train/StorageFurniture-44781-link_2-handle_6-joint_2-handlejoint_6"),
    #20

    Path("assets/drawer/train/StorageFurniture-45687-link_1-handle_1-joint_1-handlejoint_1"),
    #Path("assets/drawer/train/StorageFurniture-49140-link_3-handle_6-joint_3-handlejoint_6"),
    Path("assets/drawer/train/StorageFurniture-45940-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/drawer/train/StorageFurniture-46123-link_1-handle_5-joint_1-handlejoint_5"),
    Path("assets/drawer/train/StorageFurniture-45801-link_3-handle_9-joint_3-handlejoint_9"),
    
    #Path("assets/drawer/train/StorageFurniture-47207-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-46537-link_1-handle_2_1-joint_1-handlejoint_2_1"),
    Path("assets/drawer/train/StorageFurniture-45243-link_3-handle_9-joint_3-handlejoint_9"),
    #Path("assets/drawer/train/StorageFurniture-48253-link_1-handle_2-joint_1-handlejoint_2"),
    #Path("assets/drawer/train/StorageFurniture-49140-link_1-handle_0-joint_1-handlejoint_0"),
    
    Path("assets/drawer/train/StorageFurniture-47296-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/drawer/train/StorageFurniture-47207-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/drawer/train/StorageFurniture-46874-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/drawer/train/StorageFurniture-47178-link_2-handle_2-joint_2-handlejoint_2"),
    #Path("assets/drawer/train/StorageFurniture-45756-link_2-handle_12-joint_2-handlejoint_12"),
    
    #Path("assets/drawer/train/StorageFurniture-47089-link_1-handle_3-joint_1-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-45243-link_1-handle_3-joint_1-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-45677-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/drawer/train/StorageFurniture-46544-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/drawer/train/StorageFurniture-48878-link_0-handle_0-joint_0-handlejoint_0"),
    #40

    Path("assets/drawer/train/StorageFurniture-48517-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/drawer/train/StorageFurniture-48876-link_1-handle_3-joint_1-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-45801-link_2-handle_6-joint_2-handlejoint_6"),
    #Path("assets/drawer/train/StorageFurniture-48253-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-45132-link_0-handle_0-joint_0-handlejoint_0"),
    
    Path("assets/drawer/train/StorageFurniture-48253-link_2-handle_4_1-joint_2-handlejoint_4_1"),
    #Path("assets/drawer/train/StorageFurniture-45290-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-45612-link_5-handle_5-joint_5-handlejoint_5"),
    Path("assets/drawer/train/StorageFurniture-48740-link_1-handle_3-joint_1-handlejoint_3"),
    Path("assets/drawer/train/StorageFurniture-45784-link_1-handle_1-joint_1-handlejoint_1"),
    
    Path("assets/drawer/train/StorageFurniture-45790-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/drawer/train/StorageFurniture-46544-link_1-handle_1-joint_1-handlejoint_1"),
    #Path("assets/drawer/train/StorageFurniture-45374-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-44853-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-47711-link_3-handle_6-joint_3-handlejoint_6"),
    
    Path("assets/drawer/train/StorageFurniture-48263-link_2-handle_3-joint_2-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-47178-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-46440-link_1-handle_3-joint_1-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-48051-link_2-handle_3-joint_2-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-46060-link_1-handle_3-joint_1-handlejoint_3"),
    #60

    Path("assets/drawer/train/StorageFurniture-45238-link_2-handle_6-joint_2-handlejoint_6"),
    #Path("assets/drawer/train/StorageFurniture-46537-link_1-handle_2-joint_1-handlejoint_2"),
    Path("assets/drawer/train/StorageFurniture-46859-link_1-handle_1-joint_1-handlejoint_1"),
    #Path("assets/drawer/train/StorageFurniture-48258-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-45427-link_1-handle_1-joint_1-handlejoint_1"),
    
    #Path("assets/drawer/train/StorageFurniture-47578-link_3-handle_3-joint_3-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-46893-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-46762-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-46549-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-47252-link_2-handle_2_1-joint_2-handlejoint_2_1"),
    
    #Path("assets/drawer/train/StorageFurniture-48258-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/drawer/train/StorageFurniture-47183-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/drawer/train/StorageFurniture-46549-link_3-handle_3-joint_3-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-48740-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-46230-link_0-handle_0-joint_0-handlejoint_0"),
    
    Path("assets/drawer/train/StorageFurniture-45146-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-40147-link_1-handle_1-joint_1-handlejoint_1"),
    #Path("assets/drawer/train/StorageFurniture-46537-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-46462-link_2-handle_2-joint_2-handlejoint_2"),
    Path("assets/drawer/train/StorageFurniture-47235-link_4-handle_4-joint_4-handlejoint_4"),
    #80

    Path("assets/drawer/train/StorageFurniture-48258-link_3-handle_9-joint_3-handlejoint_9"),
    Path("assets/drawer/train/StorageFurniture-48740-link_2-handle_6-joint_2-handlejoint_6"),
    Path("assets/drawer/train/StorageFurniture-45622-link_2-handle_6_1-joint_2-handlejoint_6_1"),
    Path("assets/drawer/train/StorageFurniture-45756-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-45092-link_1-handle_1-joint_1-handlejoint_1"),
    
    Path("assets/drawer/train/StorageFurniture-48491-link_3-handle_6-joint_3-handlejoint_6"),
    Path("assets/drawer/train/StorageFurniture-45092-link_3-handle_3-joint_3-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-46443-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-46549-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/drawer/train/StorageFurniture-46130-link_1-handle_2_1-joint_1-handlejoint_2_1"),
    
    #Path("assets/drawer/train/StorageFurniture-46544-link_0-handle_0-joint_0-handlejoint_0"),
    Path("assets/drawer/train/StorageFurniture-46741-link_1-handle_1-joint_1-handlejoint_1"),
    Path("assets/drawer/train/StorageFurniture-45948-link_5-handle_15-joint_5-handlejoint_15"),
    #Path("assets/drawer/train/StorageFurniture-45135-link_1-handle_3-joint_1-handlejoint_3"),
    #Path("assets/drawer/train/StorageFurniture-46060-link_0-handle_0-joint_0-handlejoint_0"),
    
    #Path("assets/drawer/train/StorageFurniture-47296-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-47578-link_2-handle_2-joint_2-handlejoint_2"),
    #Path("assets/drawer/train/StorageFurniture-44817-link_1-handle_1-joint_1-handlejoint_1"),
    #Path("assets/drawer/train/StorageFurniture-48010-link_0-handle_0-joint_0-handlejoint_0"),
    #Path("assets/drawer/train/StorageFurniture-46762-link_2-handle_6-joint_2-handlejoint_6"),
]

# global
task_names = ["test", "FrankaPoseCabinetBase", "FrankaPoseCabinetPC"]
algo_names = ["rdm", "ppo_pn" , "ppo", "heuristics", "sac", "pregrasp_ppo", \
    "pregrasp_ppo_pn", "imitation_learning", "collect_data", "behavior_cloning", \
        "sac_il", "dagger", "dagger_ppo", "ILAD"]
no_training_algos = ["rdm", "heuristics"]

# vec_task:
clip_actions = 3.0 
clip_observations = 5.0

# base env
plane_params_static_friction = 0.1
plane_params_dynamic_friction = 0.1
control_ik_damping = 0.05

# camera
cam1_pos = gymapi.Vec3(0.5,0,2.0)
cam1_rot = gymapi.Vec3(-2.0,0., -0.8)
cam2_pos = gymapi.Vec3(0.0,0.8,2.3)
cam2_rot = gymapi.Vec3(-3.0,-2.0, -0.2)
cam3_pos = gymapi.Vec3(0.0,-0.8,2.3)
cam3_rot = gymapi.Vec3(-3.0,2.0, -0.2)
video_cam_pos = gymapi.Vec3(0.8,-1.5, 2.2)
video_cam_rot = gymapi.Vec3(-2.8, 2.2, -0.1)
general_cam_pos = gymapi.Vec3(0, 0, 10)
general_cam_rot = gymapi.Vec3(-1, 0, -1)

# simulator
sim_params_dt = 1./60.
sim_params_physx_solver_type = 1
sim_params_physx_num_position_iterations = 8
sim_params_physx_num_velocity_iterations = 8
sim_params_physx_num_threads = 8
sim_params_physx_max_gpu_contact_pairs = 8 * 1024 * 1024
sim_params_physx_rest_offset = 0.0
sim_params_physx_bounce_threshold_velocity = 0.2
sim_params_physx_max_depenetration_velocity = 1000.0
sim_params_physx_default_buffer_size_multiplier = 5.0
sim_params_physx_contact_offset = 1e-3


# robot
asset_options_fix_base_link = True
asset_options_disable_gravity = True
asset_options_flip_visual_attachments = True
asset_options_armature = 0.01
asset_options_thickness = 0.001
# robot initial pose

initial_franka_pose_r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
initial_franka_pose_p_close_drawer = gymapi.Vec3(1.2, 0.0, 0.7)
initial_franka_pose_p_close_door = gymapi.Vec3(1.2, 0.0, 0.7)
initial_franka_pose_p_open_drawer = gymapi.Vec3(0.8, 0.0, 0.5)
initial_franka_pose_p_open_door = gymapi.Vec3(0.8, 0.0, 0.4)

# asset
asset_options_fix_base_link_object = True
asset_options_disable_gravity_object = True
asset_options_collapse_fixed_joints_object = False # Merge links that are connected by fixed joints.
# asset init pose
object_init_pose_p_np = np.array([-1, 0, 1.4])
object_init_pose_r_np = np.array([0.,0.,1.,0.])
object_init_pose_p = gymapi.Vec3(-1, 0, 1.4)
object_init_pose_r = gymapi.Quat(0.,0.,1.,0.)

# numpy formatting
# edgeitems=30
# infstr="inf"
# linewidth=4000
# nanstr="nan"
# precision=2
# suppress=False
# threshold=1000
# formatter=None

COLOR20 = np.array(
                [[0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
                [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
                [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190]])

# [0.2500, 0.2499, 0.2500, 0.2500, 0.2498, 
# 0.2503, 0.8515, 0.2508, 0.2500, 0.7022, 

# 0.9294, 0.2443, 0.2387, 0.5189, 0.2889, 
# 0.2500, 0.2246, 0.6357, 0.6142, 0.2501, 

# 0.2503, 0.2255, 0.2501, 0.2500, 0.2500, 
# 1.1107, 0.2500, 0.2501, 0.2271, 0.5886, 

# 0.2500, 0.2504, 0.2498, 0.2500, 0.2287, 
# 0.6647, 0.7407, 0.4855, 0.2500, 0.2502, 

# 0.2503, 0.2474, 0.2320, 0.5151, 0.8075, 
# 0.2501, 0.7382, 0.2460, 0.2500, 0.2499, 

#   7, 10, 11, 13, 14, 
#   15, 16, 17, 18, 19,
#   22,26,29, 30,35, 
#   36, 37, 38, 43, 44, 
#   45, 47, 

# 0.2499, 0.3093, 0.9963, 0.2500, 0.2499, 
# 0.2691, 0.8527, 0.3197, 0.9649, 0.9146, 

# 0.2523, 0.2296, 0.2502, 0.9121, 0.2499, 
# 0.2501, 0.6286, 0.9111, 0.6688, 0.2493, 

# 0.2283, 0.2496, 0.2499, 0.9056, 0.8047, 
# 0.2501, 0.2503, 0.3749, 0.2497, 0.2498, 

# 0.2503, 0.2504, 0.2499, 0.2496, 0.9591, 
# 0.2500, 0.2501, 1.0929, 0.9107, 0.2502, 

# 0.7643, 0.2500, 0.2492, 1.0653, 0.6990, 
# 0.2597, 0.2314, 0.6788, 0.7360, 0.2340]
  

# #  52 53 56 57 58 
# #  59 50 62 64 66 
# #  67 68 69 71 74 
# #  75 78 85 88 89 
# #  91 94 95 96 97 
# #  98 99 100 