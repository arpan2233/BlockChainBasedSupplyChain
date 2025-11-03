# blockchain.py
import hashlib
import json
import time
import os
from typing import List, Dict

class Block:
    def __init__(self, index:int, timestamp:float, data:Dict, previous_hash:str, nonce:int=0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self, persist_file="chain.json", difficulty=2):
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.persist_file = persist_file
        # load or create genesis
        if os.path.exists(self.persist_file):
            try:
                self.load_chain()
            except Exception:
                self.create_genesis_block()
        else:
            self.create_genesis_block()

    def create_genesis_block(self):
        genesis_data = {"type":"GENESIS", "note":"genesis block"}
        genesis = Block(0, time.time(), genesis_data, "0")
        # enforce PoW for genesis to set hash
        self.chain = [genesis]
        self.save_chain()

    def proof_of_work(self, block: Block):
        computed = block.compute_hash()
        # simple PoW: find nonce where hash starts with difficulty zeroes
        while not computed.startswith('0' * self.difficulty):
            block.nonce += 1
            computed = block.compute_hash()
        block.hash = computed
        return computed

    def add_block(self, data: Dict):
        prev = self.chain[-1]
        new_index = prev.index + 1
        block = Block(new_index, time.time(), data, prev.hash)
        self.proof_of_work(block)
        self.chain.append(block)
        self.save_chain()
        return block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i-1]
            if curr.previous_hash != prev.hash:
                return False
            if curr.compute_hash() != curr.hash:
                return False
        return True

    def to_list(self):
        return [{
            "index": b.index,
            "timestamp": b.timestamp,
            "data": b.data,
            "previous_hash": b.previous_hash,
            "nonce": b.nonce,
            "hash": b.hash
        } for b in self.chain]

    def save_chain(self):
        try:
            with open(self.persist_file, "w") as f:
                json.dump(self.to_list(), f, indent=2)
        except Exception as e:
            print("Error saving chain:", e)

    def load_chain(self):
        with open(self.persist_file, "r") as f:
            raw = json.load(f)
        chain = []
        for item in raw:
            b = Block(item["index"], item["timestamp"], item["data"], item["previous_hash"], item.get("nonce", 0))
            b.hash = item.get("hash", b.compute_hash())
            chain.append(b)
        self.chain = chain

    def get_all_events(self) -> List[Dict]:
        # returns all events (skip genesis)
        events = []
        for b in self.chain[1:]:
            events.append({
                "block_index": b.index,
                "timestamp": b.timestamp,
                "data": b.data,
                "hash": b.hash,
                "previous_hash": b.previous_hash
            })
        return events
