"""
labeller.py
Brian Wang, bhw45@cornell.edu

Module for the manual annotator application.
Allows the user to view a point cloud, visualize ground plane removal + clustering results,
remove any incorrect clusters, and add any missed clusters.

Currently this is somewhat clunky to use - user needs to switch between visualizing/selecting points in the open3D
visualizer window, and entering commands on the terminal.

In the future, planning to switch this over to the new open3D GUI system - waiting on the open3d team to release
documentation on how to make GUIs.

"""
import open3d as o3d
import numpy as np
# from pointcloud.dataset import Dataset
from pointcloud.clustering import TreePointCloud
from pathlib import Path

KEY_KEEP = 'k'
KEY_RESELECT = 'r'
KEY_DELETE = 'd'
KEY_ADD = 'a'
KEY_QUIT = 'q'
KEY_VIEW = 'v'
KEY_UNDO = 'u'

# REVIEW_CLUSTER_COMMANDS = [KEY_KEEP, KEY_RESELECT, KEY_DELETE]
REVIEW_CLUSTER_COMMANDS = [KEY_KEEP, KEY_DELETE]
ADD_CLUSTER_COMMANDS = [KEY_ADD, KEY_QUIT, KEY_UNDO, KEY_VIEW]

HIGHLIGHT_PURPLE = np.array([0.8, 0.15, 0.8])  # bright purple
HIGHLIGHT_GREEN = np.array([0.1, 0.9, 0.1])
HIGHLIGHT_BLUE = np.array([0.4, 0.55, 0.85])
HIGHLIGHT_GRAY = np.array([0.6, 0.6, 0.6])


def test_calback(vis):
    selected = vis.get_picked_points()
    if len(selected) > 0:
        print(len(selected))
    return True

class Labeller(object):

    def __init__(self, dataset):
        """

        Parameters
        ----------
        dataset : Dataset
        """
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()

        # Clusters are stored as a dict, where the key is the int index of the cluster,
        # and values are a list of point indices belonging to that cluster
        self.clusters = {}
        self.dataset = dataset

        # Add the open3D point cloud to the visualizer
        # self.vis.add_geometry(dataset.global_pointcloud.get_o3d())
        # self.vis.register_animation_callback(test_calback)
        # self.vis.run()

        # while True:
            # self.update_vis()
        self.n_auto_correct = 0
        self.n_auto_incorrect = 0
        self.n_user_added = 0


    # def update_vis(self):
    #     self.vis.poll_events()
    #     self.vis.update_renderer()

    def run(self):
        self.auto_cluster()
        print("Review auto-clustering results")
        self.review_clusters()
        print("Finished reviewing clusters from auto-clustering.")
        print("Next, add new clusters if needed.")
        self.add_clusters()
        print("Finished adding new clusters. Labelling done!")

    def auto_cluster(self):
        """
        Run automated clustering on the point cloud.
        These clusters will later be reviewed by the user.

        Returns
        -------

        """
        print("[INFO] Removing ground and clustering point cloud. This may take some time...")
        clusters, cluster_indices = self.dataset.cluster_global_pointcloud()
        print("[INFO] Finished clustering")
        for i, indices in enumerate(cluster_indices):
            # Begin cluster numbering from 1
            self.clusters[i+1] = indices

    def review_clusters(self):
        final_clusters = {}
        next_cluster = 1
        pcd: o3d.geometry.PointCloud = self.dataset.global_pointcloud.get_o3d()
        ground_plane_pcd = self.dataset.global_pointcloud.ground_plane
        # Darken the ground plane
        # ground_plane_pcd.colors = o3d.utility.Vector3dVector(np.asarray(ground_plane_pcd.colors) * 0.05)

        pcd_vis = o3d.geometry.PointCloud(pcd)
        colors_vis = np.asarray(pcd_vis.colors)

        for c in self.clusters:
            cluster_indices = self.clusters[c]

            colors_vis[cluster_indices,:] = HIGHLIGHT_PURPLE
            pcd_vis.colors = o3d.utility.Vector3dVector(colors_vis)

            # Move visualizer viewpoint to this cluster
            # Viewpoint changes to directly above the cluster centroid
            # centroid = np.mean(pcd_points[self.clusters[c], :], axis=0)

            # view control cannot be changed with draw_geometries in open3d 0.10.
            # in open3d 0.11, draw_geometries gets inputs which allow moving the camera
            # view = vis.get_view_control()
            # view.set_front(centroid + [0, 0, view_distance])
            # view.set_lookat(centroid)
            # view.set_front([1.0, 1.0, 0])
            # view.set_up([0, 0, 1.0])
            # view.set_zoom(0.3)

            print("Review highlighted cluster in the visualizer window. Press Q (with the visualizer window active) when done.")

            #o3d.visualization.draw_geometries([pcd_vis, ground_plane_pcd])

            # Prompt user for action
            #command = self.get_user_command_review_clusters(c)

            command = KEY_KEEP

            if command == KEY_DELETE:
                # Skip adding this cluster to the new clusters dict
                print("Deleting cluster")
                colors_vis[cluster_indices,:] = HIGHLIGHT_GRAY
                self.n_auto_incorrect += 1
            else: # command == KEY_KEEP
                print("Keeping cluster.")
                final_clusters[next_cluster] = self.clusters[c]
                next_cluster += 1
                colors_vis[cluster_indices,:] = HIGHLIGHT_BLUE
                self.n_auto_correct += 1
        self.clusters = final_clusters

    def add_clusters(self):
        next_cluster = max([i for i in self.clusters.keys()]) + 1
        # Get the current point cloud, with ground plane removed
        pcd: o3d.geometry.PointCloud = self.dataset.global_pointcloud.get_o3d()
        ground_plane_pcd = self.dataset.global_pointcloud.ground_plane

        # Paint points that are already clustered
        while True:
            # Paint existing clusters blue
            pcd_vis = o3d.geometry.PointCloud(pcd)
            colors_vis = np.asarray(pcd_vis.colors)
            for cluster_indices in self.clusters.values():
                # colors_vis[cluster_indices, :] = HIGHLIGHT_BLUE
                colors_vis[cluster_indices, :] = np.random.random(3)
            pcd_vis.colors = o3d.utility.Vector3dVector(colors_vis)

            # Get command from user
            command = self.get_user_command_add_clusters()
            if command == KEY_ADD:
                # Select points and add a new cluster
                selected = self.select_cluster(pcd_vis)
                # Remove any selected points which belong to another cluster
                for cluster_indices in self.clusters.values():
                    selected = [i for i in selected if not i in cluster_indices]
                if len(selected) == 0:
                    print("No points selected, will not add cluster")
                else:
                    self.clusters[next_cluster] = selected
                    # Increment number for next cluster
                    next_cluster += 1
                    self.n_user_added += 1
            elif command == KEY_UNDO:
                next_cluster -= 1
                self.clusters.pop(next_cluster)
                print("Removed cluster %d" % next_cluster)
                self.n_user_added -= 1
            elif command == KEY_VIEW:
                print("Visualizing point cloud. Press 'Q' to exit visualizer.")
                o3d.visualization.draw_geometries([pcd_vis, ground_plane_pcd])
            else: # command == KEY_QUIT
                break

    def write_clusters(self):
        # lines = []
        # for c in self.clusters:
        #     point_indices = self.clusters[c]
        #     for p in point_indices:
        #         lines.append("%d %d\n" % (p, c))
        # with open(clusters_filename, 'w') as clusters_file:
        #     clusters_file.writelines(lines)
        # print("Wrote %d lines to file '%s'" % (len(lines), clusters_filename))
        clusters_filename = self.dataset.clusters_path
        npz_dict = {}
        points = np.asarray(self.get_dataset_pcd().points)
        for c in self.clusters:
            cluster_indices = self.clusters[c]
            cluster_points = points[cluster_indices,:]
            npz_dict[str(c)] = cluster_points
        print("Saved %d clusters to file %s" % (len(npz_dict), clusters_filename))
        np.savez_compressed(clusters_filename, **npz_dict)

    def write_stats(self, filename):
        lines = []
        lines.append("Correct auto-generated clusters: %d\n" % self.n_auto_correct)
        lines.append("Incorrect auto-generated clusters: %d\n" % self.n_auto_incorrect)
        lines.append("User-added clusters: %d\n" % self.n_user_added)
        with open(filename, 'w') as stats_file:
            stats_file.writelines(lines)

    def get_dataset_pcd(self):
        return self.dataset.global_pointcloud.get_o3d()


    @staticmethod
    def get_user_command_review_clusters(cluster_index=-1):
        print("----------------------------------")
        if cluster_index is not -1:
            print("[Cluster %d]" % cluster_index)
        print("Enter one of the following commands:")
        print("(%s) Keep the cluster as-is, no need for any changes." % KEY_KEEP)
        # print("(%s) Re-select the points for this cluster. "
        #       "Use this option if auto-clustering made some mistakes in selecting points." % KEY_RESELECT)
        print("(%s) Delete this cluster. "
              "Use this option if auto-clustering made a mistake." % KEY_DELETE)
        while True:
            user_input = input("Enter command [Default: 'k'eep]: ")
            if len(user_input) == 0:
                user_input = KEY_KEEP
            command = user_input[0].lower()
            if command not in REVIEW_CLUSTER_COMMANDS:
                print("Invalid command '%s'. Try again." % command)
            else:
                break
        return command

    @staticmethod
    def prompt_reselect():
        print("Select points for ")

    @staticmethod
    def get_user_command_add_clusters():
        print("----------------------------------")
        print("Enter one of the following commands:")
        print("(%s) Add a new cluster and select points for it." % KEY_ADD)
        print("(%s) Visualize the point cloud and current clusters." % KEY_VIEW)
        print("(%s) Undo, remove last added cluster." % KEY_UNDO)
        print("(%s) Finish adding new clusters and quit. " % KEY_QUIT)
        while True:
            user_input = input("Enter command: ")
            if len(user_input) > 0:
                command = user_input[0].lower()
            else:
                command = user_input
            if command not in ADD_CLUSTER_COMMANDS:
                print("Invalid command '%s'. Try again." % command)
            else:
                break
        return command

    @staticmethod
    def select_cluster(pcd):
        """
        Prompt the user to select points in the visualizer window.

        Reference: http://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html
        """

        print("")
        print(
            "1) Hold down [shift + left click] to drag a box around points"
        )
        print("   You may need to rotate the viewpoint and repeat this a few times to select all the points.")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return [pt.index for pt in vis.get_picked_points()]

