// ============================================================================================================
// UiTheme: THE single place where the headset UI gets its look.
//
// README (restyling)
//   * Every colour, size, line width, spacing, font, sprite, animation time and text format used by the UI views
//     (SourceMarkerView, EdgeArrowView, RadarRingView, StateBannerView, AlertBannerView, GlassesPanelView, ReticleView)
//     comes from an instance of this class. The views and CocktailUI contain layout and logic only.
//   * Default asset: Assets/Cocktail/UI/Resources/CocktailUiTheme.asset (created by the menu Cocktail > Add UI Components,
//     or Cocktail > Create Default UI Theme). Duplicate it, edit it in the Inspector, and drag the copy into
//     CocktailUI > Theme. Restyle = edit this asset; no script changes needed.
//   * Colours are stored as named ROLES. Defaults come from the team's neon-city palette:
//       plates    #0A0A14 (near-black, 85% opacity)   plateAlt #12101E   navy #172846 / #0D1B33   plum #510D3F   muted teal #496E73
//       accents   cyan #00E7FF  hot pink #FF44CC  deep pink #FF0090  yellow #FFE040  orange #FF6600  acid lime #CCFF00
//       text      off-white #F4F1EC (the one neutral) on the dark plates; near-black #0A0A14 on orange/light fills.
//     Roles: speech = cyan, music/other sounds = hot pink, SELECTED target + lock-on brackets + gaze highlight = acid lime,
//     alerts = orange with yellow hazard stripes, last seen / lost = muted teal (dimmed) with a deep-pink flicker when the
//     signal is lost, plate outlines = navy (accent when active). Remap a role by changing its colour.
//     Accents are for rules, brackets, pips and fills, not for body text (text is off-white on the dark plates).
//   * Design language (anime / video-game scanner HUD, expressed with shapes, not effects): chamfered plates, corner
//     brackets that snap onto a source, a thin callout line to a small data box, a radial scanner ring showing every source
//     by bearing, uppercase condensed labels, short data lines (BRG +25, LVL -31 DB, TRK / LST / LOST). Glow is used ONLY on
//     the selected target and on alerts. The only animation besides easing is the lock-on snap and a 2-frame flicker on
//     state changes / signal loss. No scanlines, aberration, shake or gradients. All flat sprites (cheap on Quest).
//   * Sprites: leave a sprite slot empty to get the built-in flat shape (chamfered plate, ring, triangle, diamond, bar).
//     Assign your own to replace it. Sprites with a 9-slice border are drawn sliced at the requested size, others are
//     stretched. Sprites are tinted by the element's colour, so use white/grey art if you want the theme colours to apply.
//   * Typeface: `font` = null uses TextMeshPro's default font asset (Liberation Sans SDF, shipped with TMP Essentials,
//     SIL Open Font License 1.1, free to embed; nothing was downloaded). Liberation Sans is not condensed, so the UI
//     condenses it horizontally with `fontCondense` (0.82 = 18% narrower, applied as a transform scale, no extra asset).
//     For a true condensed face, get an open-licence one (for example Barlow Condensed, Saira Condensed or Rajdhani, all SIL
//     OFL; check the licence on the font's page), create a TMP font asset from it (Window > TextMeshPro > Font Asset Creator),
//     assign it to `font` and set `fontCondense` to 1. `monospaceEm` forces equal digit width with TMP's <mspace> tag.
//     Default text formats are plain ASCII (no glyph the font might lack). Decorative glyphs (e.g. katakana) are
//     intentionally not used; if you add some, verify the font asset contains them and never make them the only meaning.
//   * Sizes are METRES in the headset's tracking space. Text "em" is the nominal glyph height; capital letters are about
//     0.7 of it. Things 2 m away should have em >= 0.085 (cap height ~ 0.06 m); head-locked elements sit ~1.1 m away and use
//     proportionally smaller numbers.
//   * Text formats use {placeholders}; see each field's tooltip.
// ============================================================================================================
using System;
using Cocktail.Protocol;
using TMPro;
using UnityEngine;

namespace Cocktail.Ui
{
    [Serializable]
    public struct LabelStyle
    {
        [Tooltip("Hub label this applies to, e.g. speech, music, alarm (case-insensitive).")] public string label;
        [Tooltip("Colour; alpha 0 = use the role default (speech = speechColor, alarm = alertColor, others = otherSoundColor).")] public Color color;
        [Tooltip("Optional display name; empty = use the hub label.")] public string displayName;
    }

    [Serializable]
    public struct StateStyle
    {
        public HubState state;
        [Tooltip("Banner headline for this state.")] public string title;
        [Tooltip("Accent colour; alpha 0 = role default (see UiTheme.StateColor).")] public Color color;
        [Tooltip("Show animated trailing dots (processing).")] public bool animated;
    }

    [CreateAssetMenu(fileName = "CocktailUiTheme", menuName = "Cocktail/UI Theme")]
    public class UiTheme : ScriptableObject
    {
        // ---------------------------------------------------------------- palette (named colours)
        [Header("Palette (neon-city)")]
        public Color nearBlack = Hex(0x0A0A14);
        public Color plateAlt = Hex(0x12101E);
        public Color navy = Hex(0x172846);
        public Color navyDark = Hex(0x0D1B33);
        public Color plum = Hex(0x510D3F);
        public Color mutedTeal = Hex(0x496E73);
        public Color cyan = Hex(0x00E7FF);
        public Color hotPink = Hex(0xFF44CC);
        public Color deepPink = Hex(0xFF0090);
        public Color yellow = Hex(0xFFE040);
        public Color orange = Hex(0xFF6600);
        public Color acidLime = Hex(0xCCFF00);
        [Tooltip("The one neutral: text on dark plates. Default off-white #F4F1EC.")] public Color offWhite = Hex(0xF4F1EC);

        // ---------------------------------------------------------------- roles (what each colour means)
        [Header("Roles (remap here)")]
        public Color speechColor = Hex(0x00E7FF);
        public Color otherSoundColor = Hex(0xFF44CC);
        [Tooltip("Selected target, lock-on brackets, gaze highlight.")] public Color selectedColor = Hex(0xCCFF00);
        [Tooltip("Alert plates.")] public Color alertColor = Hex(0xFF6600);
        [Tooltip("Hazard stripes on alerts.")] public Color hazardColor = Hex(0xFFE040);
        [Tooltip("Last seen / lost sources (dimmed).")] public Color lostColor = Hex(0x496E73);
        [Tooltip("Brief flicker colour when a signal is lost or the target is lost.")] public Color lostFlickerColor = Hex(0xFF0090);
        [Tooltip("Plates (alpha = opacity). Default #0A0A14 at 85%.")] public Color plateColor = Hex(0x0A0A14, 0.85f);
        [Tooltip("Plate outline when idle. Default navy #172846.")] public Color outlineColor = Hex(0x172846);
        [Tooltip("Text on dark plates.")] public Color textLight = Hex(0xF4F1EC);
        [Tooltip("Text on orange / light fills.")] public Color textDark = Hex(0x0A0A14);

        [Header("Colour usage")]
        [Range(0f, 1f)] public float secondaryTextAlpha = 0.82f;
        [Range(0f, 1f)] public float barBackgroundAlpha = 0.28f;
        [Tooltip("Alpha multiplier for a source that is tracking / last seen / lost.")]
        public float alphaTracking = 1f, alphaLastSeen = 0.8f, alphaLost = 0.55f;
        [Tooltip("Brackets and dots of sources that are not selected or gazed are dimmed to this alpha.")] [Range(0f, 1f)] public float dimAlpha = 0.6f;
        public LabelStyle[] labels =
        {
            new LabelStyle { label = "speech" },
            new LabelStyle { label = "music" },
            new LabelStyle { label = "alarm" },
        };

        // ---------------------------------------------------------------- fonts / text
        [Header("Fonts and text scale")]
        [Tooltip("TMP font asset for all text. Null = TextMeshPro default (Liberation Sans SDF, OFL).")] public TMP_FontAsset font;
        public FontStyles titleStyle = FontStyles.Bold;
        public FontStyles bodyStyle = FontStyles.Normal;
        [Tooltip("Render labels in capitals (TMP UpperCase).")] public bool uppercase = true;
        [Tooltip("Horizontal squeeze of all text (1 = none). Fakes a condensed face with the default font.")] [Range(0.5f, 1f)] public float fontCondense = 0.82f;
        [Tooltip("TMP character spacing (em/100).")] public float characterSpacing = 2f;
        [Tooltip("Monospace advance (em) for numeric readouts via <mspace>; 0 = off.")] public float monospaceEm = 0.55f;
        [Tooltip("TextMeshPro fontSize that gives 1 m nominal glyph height (TMP world text is fontSize * 0.1 m, so 10).")]
        public float fontSizePerMetre = 10f;

        // ---------------------------------------------------------------- sprites (all optional)
        [Header("Sprites (empty = built-in flat shape)")]
        public Sprite plateSprite;
        public Sprite outlineSprite;
        public Sprite barSprite;
        public Sprite bracketSprite;
        public Sprite arrowSprite;
        public Sprite reticleSprite;

        // ---------------------------------------------------------------- world markers
        [Header("Source markers (world-anchored)")]
        [Min(0.5f)] public float markerDistance = 2f;
        [Range(1, 12)] public int maxSources = 8;
        [Tooltip("Data box (w,h) shown for the selected / gazed source, metres.")] public Vector2 plateSize = new Vector2(0.66f, 0.40f);
        public float platePadding = 0.035f;
        public float accentRuleWidth = 0.012f;
        public float titleEm = 0.09f, dataEm = 0.075f, labelEm = 0.08f;
        public float levelBarHeight = 0.022f;
        [Tooltip("Short label plate under an unselected source.")] public Vector2 labelPlateSize = new Vector2(0.34f, 0.11f);
        [Tooltip("Dim bracket size of an unselected in-view source, dot size of an out-of-view one.")] public float smallBoxSize = 0.18f, dotSize = 0.045f;
        [Tooltip("Distance from the bracket box to the data box (callout line length) and gap above the label plate.")] public float calloutLength = 0.12f, labelGap = 0.03f;
        public Vector2 defaultBoxSize = new Vector2(0.40f, 0.50f);
        public Vector2 minBoxSize = new Vector2(0.22f, 0.22f), maxBoxSize = new Vector2(0.8f, 1.0f);
        [Tooltip("Bracket line width and arm length (fraction of the shorter box side).")] public float bracketThickness = 0.010f, bracketArm = 0.28f;
        public float bracketThicknessSelected = 0.022f;
        [Tooltip("Callout line width.")] public float calloutWidth = 0.006f;
        [Tooltip("Tick marks on the bracket box edges and the reticle ring.")] public float tickLength = 0.03f, tickThickness = 0.005f;
        [Tooltip("Soft halo behind the SELECTED target and ALERTS only: extra size around the shape and its alpha.")]
        public float glowSpread = 0.07f; [Range(0f, 1f)] public float glowAlpha = 0.28f;
        [Tooltip("Vertical distance between markers whose bearings are within stackWithinDeg.")] public float stackSpacing = 0.40f;
        public float stackWithinDeg = 14f;
        [Tooltip("Level meter range, dB: level_db <= min draws empty, >= max draws full.")] public float levelMinDb = -60f, levelMaxDb = 0f;
        [Tooltip("Lock-on: brackets start this much larger and ease in when a source is selected / gazed.")] public float lockOnScale = 1.7f;

        [Header("Camera box mapping (approximate)")]
        [Tooltip("Used when the camera hfov is unknown.")] public float fallbackCameraHfovDeg = 80f;
        [Tooltip("Camera frame width / height used to turn the horizontal fov into a vertical one.")] public float cameraAspect = 1.3333f;

        // ---------------------------------------------------------------- speaker-only HUD
        [Header("Speaker-only HUD (corner outline on whoever is speaking; dot at the edge if out of view; arrow toward the noise)")]
        [Tooltip("Start in the speaker-only HUD (A on the right controller cycles speaker-only, full, glasses). Off = start in the full HUD.")] public bool speakerOnlyHud = true;
        public Color speakerColor = Hex(0x00E7FF);
        [Tooltip("Thin dark line around the outline, dot and arrow so they read on light walls.")] public Color speakerOutlineColor = Hex(0x0A0A14);
        [Tooltip("Outline box around a speaker when the camera gives no box for them: full angle in degrees (width, height).")] public Vector2 speakerBoxDeg = new Vector2(22f, 46f);
        [Tooltip("Corner bracket line width, arm length (fraction of the shorter side) and the dark outer line, in metres at the marker distance.")] public float speakerBracketThickness = 0.006f, speakerBracketArm = 0.22f, speakerOutlineExtra = 0.003f;
        [Tooltip("A source counts as speaking when labelled speech, tracking, and at least this loud (dB). -55 matches the hub's own silence floor (hub/track/tracker.py active_floor_db).")] public float speakingMinDb = -55f;
        [Tooltip("Keep the outline this long after the hub stops reporting speech, so it does not flicker.")] public float speakerHoldSeconds = 0.6f;
        [Tooltip("The speaker is in view when within this many degrees of straight ahead; beyond that a dot is shown at the screen edge.")] public float speakerViewHalfDeg = 45f, speakerViewHalfVertDeg = 28f;
        [Tooltip("Speaker picker: head-locked offset, row width and height, number size (metres / em).")] public Vector3 speakerListOffset = new Vector3(0.42f, -0.2f, 1.1f);
        [Tooltip("Pointer: dot diameter and ray width in metres.")] public float pointerDotSize = 0.022f, pointerLineWidth = 0.002f;
        [Tooltip("Demo/test panel: head-locked offset of its chip (the panel opens below it).")] public Vector3 demoPanelOffset = new Vector3(-0.42f, 0.34f, 1.1f);
        [Tooltip("Numbers on the picker rows are for the team while testing; turn off for wearers (rows are then colour only).")] public bool speakerListShowNumbers = true;
        // ---- the Cocktail look (from the design mock; every size in degrees at the distance it is drawn at: boxes at markerDistance, picker at 1.1 m)
        [Header("Cocktail look")]
        public Color lookInk = Hex(0x08242F), lookPetrol = Hex(0x0A3341), lookPaper = Hex(0xF4F1E8), lookFocus = Hex(0x54899E);
        [Tooltip("Source colours 1 to 5 (acid yellow, coral, mint, haze blue, lilac). Later numbers take the midpoint of two neighbours.")]
        public Color[] lookSources = { Hex(0xD8E018), Hex(0xE88068), Hex(0x38E0B0), Hex(0x4A9EC4), Hex(0xB08CB8) };
        public bool lookFullBox = true;   // false = corner brackets
        [Range(0f, 1f)] public float lookTint = 0.25f, lookBoxLine = 0.80f, lookHairline = 0.65f, lookTab = 0.80f, lookPlate = 0.62f, lookBlock = 0.65f, lookBlockChosen = 0.90f, lookName = 0.80f, lookHoverBand = 0.55f, lookFrame = 0.90f, lookChevron = 0.70f;
        public float lookBoxLineDeg = 0.14f, lookHairlineDeg = 0.06f, lookBoxPadDeg = 0.30f, lookTabHeightDeg = 1.15f, lookTabPadDeg = 0.25f, lookTabGapDeg = 0.30f, lookTabNumDeg = 0.80f, lookTabNameDeg = 0.95f;
        public float lookRowDeg = 1.30f, lookRowGapDeg = 0.20f, lookPadDeg = 0.26f, lookBlockWDeg = 2.20f, lookNameColDeg = 4.40f, lookNameGapDeg = 0.35f, lookNumDeg = 0.80f, lookNameDeg = 1.00f;
        public float lookChamferDeg = 0.62f, lookFrameChamferDeg = 0.72f, lookBandChamferDeg = 0.50f, lookFrameOutsetDeg = 0.13f, lookBandOutsetDeg = 0.08f, lookFrameLineDeg = 0.10f;
        [Tooltip("Motion, seconds. All on one ease: 1 - 2^(-10t).")] public float tAppear = 0.28f, tDisappear = 0.42f, tCross = 0.42f, tGlide = 0.22f, tFlash = 0.60f;
        /// <summary>Size in metres of an angle in degrees at a distance (no lower limit, unlike UiMath.AngleToSize).</summary>
        public static float Deg(float deg, float distance) => 2f * distance * Mathf.Tan(0.5f * deg * Mathf.Deg2Rad);

        public float speakerListWidth = 0.1f, speakerListRowHeight = 0.05f, speakerListEm = 0.04f;
        [Tooltip("Tracked hands are drawn as dots on the joints joined by thin lines: colour (alpha = opacity), dot diameter and line width in metres.")] public Color handDotColor = new Color(0.96f, 0.95f, 0.93f, 0.7f);
        public float handDotSize = 0.005f, handLineWidth = 0.0015f;
        [Tooltip("Draw the hand skeleton (dots and lines). Off: only your real hands show through the passthrough.")] public bool showHandDots = true;
        [Tooltip("Developer tools: DEMO panel, test cubes and sound, fake-data fallback, diagnostics text, event log. Off in normal use; B + Y pressed together toggles it on the headset.")] public bool devMode = false;
        [Tooltip("Keep the small DEV chip (upper left) on screen so DevMode can be switched on and off with a pinch. Turn off for the demo.")] public bool showDevToggle = true;
        [Tooltip("Opacity of the colour layer over a speaker you are pointing at (0 to 1).")] [Range(0f, 1f)] public float speakerHoverTint = 0.28f;
        [Tooltip("Dot size, and the dashed uncertainty ring (small ring = sure, large ring = unsure), in metres.")] public float speakerDotSize = 0.05f, speakerRingMin = 0.14f, speakerRingMax = 0.34f;

        // ---------------------------------------------------------------- unlocated chips
        [Header("Unlocated sources (no bearing)")]
        public Vector3 chipRowOffset = new Vector3(0f, -0.50f, 1.1f);
        public float chipScale = 0.55f, chipSpacing = 0.40f;
        public int maxChips = 3;

        // ---------------------------------------------------------------- radar ring
        [Header("Radar ring (head-locked scanner: every source by bearing, 0 = ahead at top)")]
        public bool showRadar = true;
        public Vector3 radarOffset = new Vector3(0f, 0f, 1.1f);
        public float radarRadius = 0.27f, radarLineWidth = 0.004f;
        [Tooltip("Tick marks every N degrees; the 90-degree ones are longer.")] public float radarTickDeg = 30f;
        public float radarTick = 0.022f, radarTickMajor = 0.04f;
        public float radarPipSize = 0.05f, radarPipSizeSelected = 0.075f;
        [Tooltip("Cone showing the field of view in which a source counts as in view (full angle, degrees) and its alpha.")] public float radarFovDeg = 80f; [Range(0f, 0.5f)] public float radarFovAlpha = 0.10f;
        [Range(0f, 1f)] public float radarAlpha = 0.9f;

        [Header("Compass strip (optional alternative; off by default)")]
        public bool showCompass = false;
        public Vector3 compassOffset = new Vector3(0f, 0.36f, 1.1f);
        public float compassWidth = 1.1f, compassSpanDeg = 180f;
        public float compassTickMinorDeg = 10f, compassTickMajorDeg = 30f;
        public float compassTickMinor = 0.016f, compassTickMajor = 0.030f, compassTickWidth = 0.003f;
        public float compassPipSize = 0.045f, compassEm = 0.042f;

        // ---------------------------------------------------------------- edge arrows
        [Header("Edge arrows (secondary, out-of-view sources)")]
        public bool showEdgeArrows = true;
        public float edgeDistance = 1.1f;
        public float edgeHalfWidthDeg = 40f, edgeHalfHeightDeg = 14f;
        public float arrowSize = 0.07f;
        public float arrowTitleEm = 0.045f, arrowDetailEm = 0.038f;
        public Vector2 arrowBoxSize = new Vector2(0.46f, 0.125f);
        public float arrowTextGap = 0.02f, arrowStackSpacing = 0.16f;

        // ---------------------------------------------------------------- state banner / HUD
        [Header("State banner (head-locked)")]
        public Vector3 bannerOffset = new Vector3(0f, -0.40f, 1.1f);
        public Vector2 bannerSize = new Vector2(0.80f, 0.20f);
        public float bannerTitleEm = 0.056f, bannerDetailEm = 0.044f;
        public float bannerRuleWidth = 0.010f;
        public StateStyle[] states =
        {
            new StateStyle { state = HubState.Listening,        title = "Listening" },
            new StateStyle { state = HubState.Selected,         title = "Selected" },
            new StateStyle { state = HubState.Playing,          title = "Playing" },
            new StateStyle { state = HubState.TargetLost,       title = "Target lost" },
            new StateStyle { state = HubState.ReplayProcessing, title = "Replay processing", animated = true },
            new StateStyle { state = HubState.ReplayPlaying,    title = "Replay playing" },
        };
        public string[] modeNames = { "Fast", "Balanced", "Accurate" };

        [Header("Glasses-style HUD (toggle)")]
        public Vector3 glassesOffset = new Vector3(0.28f, -0.18f, 1.1f);
        public Vector2 glassesSize = new Vector2(0.50f, 0.20f);
        public float glassesArrowSize = 0.10f, glassesTitleEm = 0.05f, glassesDetailEm = 0.042f;

        [Header("Alerts (head-locked)")]
        public Vector3 alertOffset = new Vector3(0f, 0.26f, 1.1f);
        public Vector2 alertSize = new Vector2(1.0f, 0.20f);
        public float alertEm = 0.06f, alertDetailEm = 0.045f, alertArrowSize = 0.12f;
        [Tooltip("Width of the diagonal hazard-stripe accent on alert plates.")] public float alertStripeWidth = 0.09f;
        public float alertSecondsCritical = 9f, alertSecondsNormal = 5f;

        [Header("Reticle (head gaze)")]
        public float reticleDistance = 2f, reticleDiameter = 0.05f, reticleHoverDiameter = 0.075f, reticleDotDiameter = 0.012f;
        [Range(0f, 1f)] public float reticleAlpha = 0.85f;

        // ---------------------------------------------------------------- interaction / animation
        [Header("Interaction and animation")]
        [Tooltip("Seconds of steady gaze that fill the dwell bar (visual only; the trigger/pinch selects).")] public float dwellSeconds = 0.6f;
        [Tooltip("Gaze must rest this long before the hover highlight (data box) appears.")] public float hoverHighlightSeconds = 0.20f;
        [Tooltip("Marker position smoothing (1/s, higher = snappier) and scale/colour easing.")] public float positionSmoothing = 7f, easing = 12f;
        public float fadeInSeconds = 0.25f;
        [Tooltip("A marker is removed this long after its source stops appearing in updates (prevents flicker).")] public float removeGraceSeconds = 1.2f;
        [Tooltip("Local 'pending select' feedback until the hub confirms.")] public float pendingSelectSeconds = 0.8f;
        public float processingDotsPeriod = 0.45f;
        [Tooltip("State-change / signal-lost flicker: number of frames and the dip alpha.")] public int flickerFrames = 2; [Range(0f, 1f)] public float flickerAlpha = 0.35f;

        // ---------------------------------------------------------------- text formats
        [Header("Text formats ({placeholders})")]
        [Tooltip("Data box and short label. {label} {conf} (percent) {brg} (signed world bearing deg) {db} (level dB) {id}")]
        public string markerTitle = "{label}", markerConf = "{conf}%", markerBearing = "BRG {brg}", markerLevel = "LVL {db} DB";
        [Tooltip("Track state codes.")] public string statusTracking = "TRK";
        [Tooltip("{age} seconds since last observed")] public string statusLastSeen = "LST {age}S";
        public string statusLost = "LOST";
        public string statusUnlocated = "NO BRG";
        public string selectedTag = "TARGET LOCK";
        [Tooltip("{label} {id}")] public string arrowTitle = "{label}";
        [Tooltip("{clock} {bearing}")] public string arrowDetail = "{clock} O'CLOCK  {bearing}";
        [Tooltip("{state}")] public string bannerTitle = "{state}";
        [Tooltip("{target} {mode}")] public string bannerDetail = "{target}  //  {mode}";
        public string noTarget = "NO TARGET";
        [Tooltip("{target} {state}")] public string glassesTitle = "{target}";
        public string glassesDetail = "{state}";
        [Tooltip("Appended to the replay state detail. {seconds}")] public string replayDetail = "REPLAY {seconds}S";
        public string degreeUnit = " DEG", aheadWord = "AHEAD", behindWord = "BEHIND", rightWord = "R", leftWord = "L";
        [Tooltip("{clock} {bearing}")] public string alertDetail = "{clock} O'CLOCK  {bearing}";

        // ---------------------------------------------------------------- derived colours
        public Color TextSecondary { get { var c = textLight; c.a *= secondaryTextAlpha; return c; } }
        public Color BarBackground { get { var c = textLight; c.a = barBackgroundAlpha; return c; } }

        /// <summary>Colour for a sound label (theme label table first, then role defaults).</summary>
        public Color LabelColor(string label)
        {
            if (labels != null && label != null)
                for (int i = 0; i < labels.Length; i++)
                    if (string.Equals(labels[i].label, label, StringComparison.OrdinalIgnoreCase) && labels[i].color.a > 0f)
                        return labels[i].color;
            if (string.Equals(label, "speech", StringComparison.OrdinalIgnoreCase)) return speechColor;
            if (string.Equals(label, "alarm", StringComparison.OrdinalIgnoreCase)) return alertColor;
            return otherSoundColor;
        }

        public string DisplayLabel(string label)
        {
            if (labels != null && label != null)
                for (int i = 0; i < labels.Length; i++)
                    if (string.Equals(labels[i].label, label, StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(labels[i].displayName))
                        return labels[i].displayName;
            return label ?? "?";
        }

        public StateStyle GetState(HubState state)
        {
            if (states != null)
                for (int i = 0; i < states.Length; i++)
                    if (states[i].state == state) return states[i];
            return new StateStyle { state = state, title = state.ToString() };
        }

        /// <summary>Accent colour of a hub state: listening = cyan, selected/playing = lime, processing = yellow, replay = hot pink, lost = deep pink.</summary>
        public Color StateColor(HubState s)
        {
            var st = GetState(s);
            if (st.color.a > 0f) return st.color;
            switch (s)
            {
                case HubState.Listening: return speechColor;
                case HubState.TargetLost: return lostFlickerColor;
                case HubState.ReplayProcessing: return hazardColor;
                case HubState.ReplayPlaying: return otherSoundColor;
                default: return selectedColor;
            }
        }

        public string ModeName(Mode m)
        {
            int i = (int)m;
            return modeNames != null && i >= 0 && i < modeNames.Length ? modeNames[i] : m.ToString();
        }

        public float FontSizeForEm(float em) => em * fontSizePerMetre;

        public static Color Hex(int rgb, float a = 1f) =>
            new Color(((rgb >> 16) & 0xFF) / 255f, ((rgb >> 8) & 0xFF) / 255f, (rgb & 0xFF) / 255f, a);
    }
}
