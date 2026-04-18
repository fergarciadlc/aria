import os
import sys
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from compute_fakeprints import fakeprint, spectrogram

TRACKS_UUIDS = [
    '098fb51f-2cda-4b3e-ad13-3ead9d0c8474', '10aba683-c3f3-47f5-a973-ec8bab1728c7', '1456630e-375e-4035-b988-316dd7613124', '1506ff5a-965b-42d7-9b30-dec23f0c7708', '16d74cda-230d-4e2d-9523-366e79e46ad5',
    '2065b20f-de5e-4832-873a-455486a69cda', '21fbb812-1dca-470c-92f7-77fcdef594bf', '26e0ca18-07ec-4f25-a666-12294dc6304c', '2d31e68c-3942-4ebe-925f-96a1d4fd8872', '39f517b8-3f65-4701-a7dd-9afb535bf280',
    '3d1fb6d0-34f9-4292-b949-73cdca7d32e3', '408e4a38-6b13-4c99-9785-f7d76eddcc07', '411eede9-caa2-4778-b514-2b158cbe0877', '462db679-bf70-4d31-a1ec-2ddb2732def6', '47bd53a3-415b-4109-b0a6-46c823951380',
    '48a1f90b-3946-42e6-9281-042dae55cf3d', '48e385d2-e879-4c7e-98c7-6b07e288d349', '4b5ec1bb-199f-4e95-94ec-38d76e3c9247', '55070996-cb72-4838-bc5d-63777793975d', '56f9bbe5-19c8-448e-8038-2e81124f7452',
    '5c680ff7-3e66-4045-a916-b466438bc252', '665127e4-3830-4168-84bb-e5683f714082', '665501bb-e0d1-4832-b45b-3b5e71959536', '69140cf1-d636-425d-8d1b-7475751e50c6', '6bda7c27-a953-4f76-86ef-ef4eec8e90e9',
    '6e55b9b4-7ae6-4423-ac5f-e5c3afb3f1e0', '7851f394-6ca7-4a8f-ab19-1e9415a87afe', '791bb3da-1297-4753-ada6-05cc2211bd0d', '79fa0de7-b8a4-486b-83c6-e3dff9346557', '82859e34-8747-4f31-b030-74f2a6944e2d',
    '87632a78-03ba-45e2-bf70-97959f2a0c54', '889dcec4-0272-4ab9-a5e7-182070f070f3', '89f4f0da-462a-4fa5-97b8-4cac3175ec05', '911ecd21-99dd-4848-b656-9cbcabb3c64c', '92508574-c071-4725-8692-5ac02cba7b1f',
    '95ef7756-07d8-4957-8c91-abda0886dbc5', 'a37c16a5-36da-40ff-bf31-fbeb87df1583', 'c05b4b3f-d500-4106-be91-a74f08c8535c', 'c44c54bc-86d6-4007-ba31-0f2afabfe48f', 'ce281a9a-281f-42f1-9596-241c9af454a0',
    'ceb821d9-7d97-430c-83db-b65b160428ef', 'd3114519-48e1-4294-8c34-54597638bfa5', 'dd12d6d8-3da5-48d6-8cd5-03082cb7d6c3', 'dd4843f8-9018-48e8-a50a-984921c2bc23', 'ded18018-f0be-48f7-b952-424f421d5deb',
    'e049f0af-3c7c-48dc-a0d5-6516b5bb9834', 'e30d586c-a5d7-4d4c-b8c9-62c4a5e1d7db', 'e6593245-97f1-4a8e-ba0d-cf1de1f59a80', 'e6d44e70-5807-4731-8f35-69157857627e', 'f3af46bc-ec7f-4c51-a076-3f213c7d4956',
    'f3b662ee-34ff-4f17-aece-76c5a600d91d', 'fb0885da-1478-4d5c-a507-e969af4f384b', 'fd6b9884-9b0a-4ad5-9ed9-1309677a8777',
]
# TRACKS_UUIDS = ["87632a78-03ba-45e2-bf70-97959f2a0c54"]


U_FOLDERS = ['U0', 'U1', 'U2', 'U3', 'U4']
BASE_PATH = '/media/daveloay/91a7c9e9-247c-4ff8-875e-36cacebd550e/DSs/aria'
OUTPUT_FOLDER = 'fakeprints'
SR = 44100
F_RANGE = [0, 5000]
MAX_WORKERS = os.cpu_count() - 2


def process_one(track_uuid, u_folder):
    path = os.path.join(BASE_PATH, track_uuid, 'mixes', 'encodec_3', u_folder)
    save_path = os.path.join(OUTPUT_FOLDER, f'fakeprints_{track_uuid}_{u_folder}.npy')

    f_list = glob(os.path.join(path, '*.mp3'))
    if not f_list:
        return track_uuid, u_folder, 0

    fakeprints = {}
    for f in f_list:
        f_name = os.path.basename(f)
        spectro = spectrogram(f, SR=SR)
        fakeprints[f_name] = fakeprint(spectro, f_range=F_RANGE, SR=SR)

    np.save(save_path, fakeprints)
    return track_uuid, u_folder, len(f_list)


if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f'Processing {len(TRACKS_UUIDS) * len(U_FOLDERS)} folders...')
    print(f"Number of workers: {MAX_WORKERS}")

    jobs = [(uuid, u) for uuid in TRACKS_UUIDS for u in U_FOLDERS]
    total = len(jobs)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, uuid, u): (uuid, u) for uuid, u in jobs}
        skipped = []
        with tqdm(total=total, desc='fakeprints') as pbar:
            for future in as_completed(futures):
                uuid, u_folder, n_files = future.result()
                if n_files == 0:
                    skipped.append((uuid, u_folder))
                pbar.set_postfix(track=uuid[:8], u=u_folder, files=n_files)
                pbar.update(1)

    if skipped:
        print(f'\n{len(skipped)} empty folders (no .mp3 files found):')
        for uuid, u in skipped:
            print(f'  {uuid} / {u}')
    print('Done!')
