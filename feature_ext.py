import os
import numpy as np
import torch
from tqdm import tqdm
import glob
import random
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
import warnings
from concurrent.futures import ProcessPoolExecutor
from collections import deque

warnings.filterwarnings("ignore")

# --- 全局配置 ---
SEQ_LEN = 0000
TARGET_SAMPLES_BENIGN =0000
TARGET_SAMPLES_ATTACK =0000


def extract_pcap_features_worker(args):
    pcap_path, label, class_name = args
    c_name_lower = class_name.lower()

    try:
        packets = rdpcap(pcap_path)
        if len(packets) < 1: return []

        timestamps = []
        sizes = []
        proto_codes = []

        recent_udp_ports = deque(maxlen=16)

        pkt_meta = []
        try:
            first_src = None
            for pkt in packets:
                if IP in pkt:
                    # 1. UDP 扫描不应包含 TCP
                    if 'scan_su' in c_name_lower and TCP in pkt:
                        continue
                    # 2. TCP 类攻击不应包含 UDP
                    if ('mqtt_bruteforce' in c_name_lower or
                        'sparta' in c_name_lower or
                        'scan_a' in c_name_lower) and UDP in pkt:
                        continue
                    # 3. 清洗 UDP 类攻击文件夹
                    if 'udp' in c_name_lower and TCP in pkt:
                        continue
                    # 4. 清洗 TCP/HTTP 类攻击文件夹
                    if ('tcp' in c_name_lower or 'http' in c_name_lower or 'theft' in c_name_lower) and UDP in pkt:
                        continue
                    # 5. OS Scan
                    if 'scan_os' in c_name_lower and UDP in pkt:
                        continue

                    if first_src is None: first_src = pkt[IP].src
                    ts = float(pkt.time)
                    direction = 1.0 if pkt[IP].src == first_src else -1.0
                    size = np.log1p(float(len(pkt))) * direction

                    p_type = 'OTHER'
                    flags_val = 0
                    dport = 0
                    plen = 0

                    if TCP in pkt:
                        p_type = 'TCP'
                        flags_val = int(pkt[TCP].flags)

                    elif UDP in pkt:
                        p_type = 'UDP'
                        dport = pkt[UDP].dport
                        plen = len(pkt[UDP].payload)

                    pkt_meta.append({
                        'ts': ts, 'size': size, 'type': p_type,
                        'flags_val': flags_val, 'dport': dport,
                        'plen': plen
                    })
        except:
            return []

        if len(pkt_meta) < 1: return []

        for meta in pkt_meta:
            timestamps.append(meta['ts'])
            sizes.append(meta['size'])

            code = 0.0

            if meta['type'] == 'TCP':
                code = float(meta['flags_val'])

            elif meta['type'] == 'UDP':
                curr_port = meta['dport']
                base_code = 0.0

                if meta['plen'] == 0:
                    base_code = 11.0  # Empty UDP
                else:
                    if len(recent_udp_ports) == 0:
                        base_code = 10.0
                    else:
                        min_diff = min([abs(curr_port - p) for p in recent_udp_ports])

                        if min_diff == 0:
                            base_code = 7.0  # Same Port (Normal/Flood)
                        elif min_diff == 1:
                            base_code = 8.0  # Sequential (Scan)
                        elif min_diff < 10:
                            base_code = 9.0  # Close Port (Scan)
                        else:
                            unique_ports = len(set(recent_udp_ports))
                            if unique_ports > 6:
                                base_code = 12.0  # High Entropy (Random Scan)
                            else:
                                base_code = 10.0  # Random New (Normal)

                    recent_udp_ports.append(curr_port)
                code = 256.0 + base_code
            proto_codes.append(code)

        # IAT 处理
        timestamps = np.array(timestamps)
        iat = np.diff(timestamps)
        iat = np.log1p(iat * 1000)
        iat = np.insert(iat, 0, 0)

        # 序列 Padding
        feats_list = [iat, sizes, proto_codes]
        processed_feats = []
        for f in feats_list:
            f = np.array(f)
            if len(f) > SEQ_LEN:
                processed_feats.append(f[:SEQ_LEN])
            else:
                processed_feats.append(np.pad(f, (0, SEQ_LEN - len(f)), 'constant'))

        feature = np.stack(processed_feats, axis=1).astype(np.float32)
        return [{'feature': feature, 'label': label, 'class_name': class_name}]

    except Exception:
        return []

def process_folder_parallel(file_list, label, class_name, save_name, target_count=None, max_workers=16):
    if not file_list: return
    data_buffer = []
    total_files = len(file_list)
    processed_idx = 0
    if target_count is None: target_count = total_files
    if total_files < target_count: target_count = total_files
    print(f"    -> Target Valid Samples: {target_count} | Total Files Available: {total_files}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while len(data_buffer) < target_count and processed_idx < total_files:
            needed = target_count - len(data_buffer)
            batch_size = int(needed * 1.3)
            batch_size = max(batch_size, 100)
            batch_files = file_list[processed_idx: processed_idx + batch_size]
            processed_idx += len(batch_files)
            if not batch_files: break
            tasks = [(f, label, class_name) for f in batch_files]
            results = list(executor.map(extract_pcap_features_worker, tasks))
            for res_list in results:
                if res_list:
                    for item in res_list:
                        item['feature'] = torch.from_numpy(item['feature'])
                        data_buffer.append(item)
    if target_count is not None and len(data_buffer) > target_count:
        data_buffer = data_buffer[:target_count]
    if len(data_buffer) > 0:
        torch.save(data_buffer, save_name)
        print(f"    -> Saved: {save_name} ({len(data_buffer)} samples)")
    else:
        print(f"    -> No valid samples found.")

def process_dataset(source_root, save_path, is_pretrain=False):
    if not os.path.exists(save_path): os.makedirs(save_path)
    print(f"Scanning directory: {source_root}")
    subfolders = sorted([f.path for f in os.scandir(source_root) if f.is_dir()])
    if not subfolders and glob.glob(os.path.join(source_root, "*.pcap")): subfolders = [source_root]
    class_map = {}
    if not is_pretrain:
        benign_folder = None
        for folder in subfolders:
            if "benign" in os.path.basename(folder).lower():
                benign_folder = folder
                break
        if benign_folder:
            class_map[os.path.basename(benign_folder)] = 0
            curr = 1
            for folder in subfolders:
                name = os.path.basename(folder)
                if name != os.path.basename(benign_folder):
                    class_map[name] = curr
                    curr += 1
        else:
            for i, folder in enumerate(subfolders): class_map[os.path.basename(folder)] = i
        print(f"Class Map: {class_map}")
    for i, folder_path in enumerate(subfolders):
        folder_name = os.path.basename(folder_path)
        print(f"[{i + 1}/{len(subfolders)}] Processing Folder: {folder_name}")
        if is_pretrain:
            label = 0
            c_name = "Pretrain_Data"
            target_limit = None
        else:
            label = class_map.get(folder_name, -1)
            c_name = folder_name
            target_limit = TARGET_SAMPLES_BENIGN if label == 0 else TARGET_SAMPLES_ATTACK
        all_fs = glob.glob(os.path.join(folder_path, "*.pcap"))
        if not all_fs: continue
        random.shuffle(all_fs)
        save_file_name = os.path.join(save_path, f"{folder_name}.pt")
        if target_limit is not None and len(all_fs) <= target_limit: target_limit = len(all_fs)
        process_folder_parallel(all_fs, label, c_name, save_file_name, target_count=target_limit)

if __name__ == "__main__":

    RAW_P = "0000"
    OUT_P = "0000"
    process_dataset(RAW_P, OUT_P, is_pretrain=True)

    RAW_B = "0000"
    OUT_B = "0000"
    process_dataset(RAW_B, OUT_B, is_pretrain=False)
