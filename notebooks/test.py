import sys; sys.path.append('../PhiFlow'); sys.path.append('../src')

from control.pde.incompressible_flow import IncompressibleFluidPDE
from control.control_training import ControlTraining
from control.sequences import StaggeredSequence
from phi.flow import *

domain = Domain([64, 64])
step_count = 16
dt = 1.0
data_path = 'shape-transitions'
test_range = range(100)

# Create fresh app instance
staggered_app = ControlTraining(
    step_count,
    IncompressibleFluidPDE(domain, dt),
    datapath=data_path,
    val_range=range(100, 200),
    train_range=range(200, 1000),
    trace_to_channel=lambda _: 'density',
    obs_loss_frames=[step_count],
    trainable_networks=['CFE', 'OP2', 'OP4', 'OP8', 'OP16'],
    sequence_class=StaggeredSequence
).prepare()

def run_simulation(checkpoints, f_name):

    staggered_app.load_checkpoints(checkpoints)


    states = staggered_app.infer_all_frames(test_range)

    print(f"Number of states: {len(states)}")
    print(f"Type of first state: {type(states[0]) if states else 'N/A'}")
    print(f"Attributes of first state: {dir(states[0]) if states else 'N/A'}")
    if states:
        print("First state's density max:", np.max(states[0].density.data))
        print("First state's density min:", np.min(states[0].density.data))


    import pylab
    batches = [0, 1, 2]
    pylab.subplots(len(batches), 9, sharey='row', sharex='col', figsize=(12, 7))
    pylab.tight_layout(w_pad=0)
    for i, batch in enumerate(batches):
        for t in range(9):
            pylab.subplot(len(batches), 9, t + 1 + i * 9)
            pylab.title('t=%d' % (t * 2))
            pylab.imshow(states[t * 2].density.data[batch, ..., 0], origin='lower')

    pylab.savefig(f_name)
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()

# Load model from saved checkpoint
# If you saved the checkpoint dict, reload from it:
# Example (update with your real paths):
staggered_checkpoint = {
    'CFE': '/home/soeren/phi/model/control-training/sim_000104/checkpoint_00001000',
    'OP2': '/home/soeren/phi/model/control-training/sim_000104/checkpoint_00001000',
    'OP4': '/home/soeren/phi/model/control-training/sim_000104/checkpoint_00001000',
    'OP8': '/home/soeren/phi/model/control-training/sim_000104/checkpoint_00001000',
    'OP16': '/home/soeren/phi/model/control-training/sim_000104/checkpoint_00001000',
}

run_simulation(staggered_checkpoint, 'triangles_1000_model.png')

staggered_checkpoint = {net: '../networks/shapes/staggered/all_53750' for net in ['CFE', 'OP2', 'OP4', 'OP8', 'OP16']}

run_simulation(staggered_checkpoint, 'triangles_prebuilt_model.png')