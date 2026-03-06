from .digs import digs_loss
from .developable import developable_loss
from .ncadr import ncadr_loss

losses = [digs_loss, developable_loss, ncadr_loss]
