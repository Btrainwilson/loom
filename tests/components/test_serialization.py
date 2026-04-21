# test_actionspace_serialization.py

import os
import tempfile
from loomlib import ActionDispatchSpace  # replace with the actual filename if needed

def test_actionspace_serialization():
    # Create and populate the original ActionSpace
    original = ActionDispatchSpace()
    original.add_types({
        "theta": [0.0, 0.1, 0.2],
        "dist": [1.0, 2.0, 3.0]
    })

    original.add_action("rotate", {"angle": "theta"})
    original.add_action("move", {"distance": "dist"})

    original.compile()

    # Temporary path for JSON file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        # Save and reload
        original.save_json(tmp_path)
        loaded = ActionSpace.load_json(tmp_path)

        # Assertions
        assert original.action_types == loaded.action_types
        assert original.types == loaded.types
        assert original.actions == loaded.actions
        assert loaded.handlers["rotate"] is None  # handlers should be None on reload

        # Roundtrip encode/decode test
        sequence = [{"rotate": {"angle": 0.1}}, {"move": {"distance": 2.0}}]
        encoded = original.encode(sequence)
        decoded = loaded.decode(encoded)
        assert decoded == [
            {"type": "rotate", "params": {"angle": 0.1}},
            {"type": "move", "params": {"distance": 2.0}}
        ]

        print("✅ Serialization test passed!")

    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_actionspace_serialization()
