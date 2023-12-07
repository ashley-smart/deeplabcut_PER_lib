# adjust keybindings to your liking
import napari

KEYMAP = {
    'p': 'proboscis_out',
    'o': 'big_proboscis',
    'l': 'light',
}

viewer = napari.Viewer()

# this writes the frame, layer source, and action each time you press a key
def on_keypress(key, viewer):
    action = KEYMAP[key]
    frame = viewer.dims.current_step[0]
    layer = viewer.layers.selection.active or viewer.layers[-1]

    show_info(action)  # if you want some visual feedback
    with open(CSV_OUT, 'a') as f:
        csv.writer(f).writerow([layer.source.path, frame, action])

for key in KEYMAP:
    viewer.bind_key(key, partial(on_keypress, key))

napari.run()