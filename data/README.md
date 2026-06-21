# Customer dataset placement

The automotive schematic PDFs are proprietary and are not included in this delivery.

Place or reference two board revisions when running the demo:

```text
data/
├── old_schematic.pdf
└── new_schematic.pdf
```

Then run:

```powershell
python demo.py --old .\data\old_schematic.pdf --new .\data\new_schematic.pdf
```

PDF files are ignored by Git by default. If the customer is permitted to publish a sanitized sample, explicitly force-add only that approved file or adjust `.gitignore`.
