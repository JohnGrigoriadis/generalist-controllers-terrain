def remove_terrain():
        """
        Remove a terrain that the controller is stuck on and continue the evolution on the remaining terrains.
        Then add the removed terrain to the bad terrains list.
        """
        global bad_terrains
        print("Removing a terrain")

        to_remove = np.argmax(terrains, axis=0)[0]
        if len(bad_terrains) == 0:
            bad_terrains = terrains[to_remove]
        else:
              bad_terrains = np.concatenate((bad_terrains, terrains[to_remove]))
        terrain_params = np.delete(terrains, to_remove, axis=0)

        return terrain_params
