def extract_spike_profile(gabor_pooling_layer_maps):
    spike_profile = []
    for map_index, gabor_pooling_layer_map in enumerate(gabor_pooling_layer_maps):
        map_height, map_width = gabor_pooling_layer_map.shape
        for i in range(map_width):
            for j in range(map_height):
                spike_profile.append((map_index, i, j, gabor_pooling_layer_map[i][j]))

    spike_profile.sort(key=lambda x: x[3], reverse=True)

    return spike_profile
